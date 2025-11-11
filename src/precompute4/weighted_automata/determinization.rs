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

/*
High-level overview of the accelerated determinization:

Key idea 1 (Transfer compilation):
- For each MacroSig s and each relevant label ℓ (including the default “None”), precompute a compact set of contributions
  T[s][ℓ] = { (target_sig, mask) } where mask is the union of all compiled-step weights that apply when starting from states
  represented by s and taking ℓ. This includes default steps (subject to exceptions and blocking by label-specific exceptions)
  and label-specific exception steps. This converts the combinatorial per-node accumulation into a simple looping over T
  with per-node gates as masks.

Key idea 2 (Per-node aggregate cache with intersection restriction):
- For each composition node N (set of MacroSig gates with weights), cache its full aggregated transitions A_N[ℓ] over all labels ℓ,
  as vectors of (target_sig, mask_N→target(ℓ)) where mask_N→target(ℓ) = OR_{s∈gates} ( gate[s] & T[s][ℓ][target] ).
- Behavioral compatibility over an intersection I reduces to checking that for every ℓ, A_N1[ℓ]∧I == A_N2[ℓ]∧I. This holds
  because AND distributes over OR. Therefore, we never recompute transitions on the fly; we only mask cached vectors.

Key idea 3 (One-shot new-target aggregation + behavior indexing):
- When creating or merging a target node for a new gates map M, compute its cache A_M once.
- Use robust fingerprints of full A_M (and a default-only fallback) to look up compatible existing nodes via indexes.
  This removes most of the linear scans over all existing nodes, yet we still verify exact equality on intersections,
  so merge efficacy is preserved.

Key idea 4 (Lazy epsilon-closure + epsilon-only future weights):
- Compute Fε[s] (zero-length acceptance) once via epsilon-only reverse propagation.
- Compute epsilon-closures only on-demand for target states that actually appear in non-epsilon edges while compiling steps.
  This replaces the O(n·E) pre-pass and can cut orders of magnitude on large graphs.

Mathematical correctness sketch:
- Final weights for determinized nodes depend only on zero-length acceptance from NWA states under epsilon paths.
  Fε[s] = final[s] ∪ ⋃_{ε s->t}(wε ∧ Fε[t]) is the least fixpoint over epsilon edges; it equals the union over all ε-paths
  of the path-weight∧final[t], which is exactly what the prior per-state epsilon-closure accumulation computed.
- Aggregation and intersection laws remain unchanged. We only accelerate candidate selection; the actual merge check still
  uses the original equal_restricted_pairs on the precise intersection mask, so we neither introduce incorrect merges nor
  miss true merges. Therefore, the number of created states due to missed opportunities does not increase.
*/

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

    // Cached aggregate transitions over full gate:
    // by_label[0] is default, by_label[li] corresponds to labels.list[li-1]
    cached_by_label: Option<NodeAggCache>,
    cache_dirty: bool,

    // Fingerprints for fast indexing (computed from cached_by_label).
    // These are safe accelerators; we still verify equality on intersections.
    behavior_fp: Option<u64>,
    default_fp: Option<u64>,
}

fn accumulate(dst: &mut HashMap<usize, Weight>, compiled: &[(usize, Weight)], gate: &Weight) {
    for (sid, w) in compiled.iter() {
        let x = w & gate;
        if !x.is_empty() {
            *dst.entry(*sid).or_default() |= &x;
        }
    }
}

/*
Transfer compilation:

- We compress labels i16 -> LabelIndex (usize). Index 0 is reserved for default (None).
- For each MacroSig s:
   def_total: union of all its default steps (step_id) compiled to by-sig pairs
   per_label[l]: union of:
       - (if label l allowed for default) those default steps' compiled pairs that do not have l in their exceptions
       - exception steps compiled pairs for label l
   block_labels: labels for which default must be blocked even if there are no explicit exception contributions
                 (ex labels and default exception labels)
*/
struct Labels {
    list: Vec<i16>,                 // Dense [0..L): labels in ascending order (Option::None handled separately)
    index: HashMap<i16, usize>,     // i16 -> idx in list
}
impl Labels {
    fn new(mut all: BTreeSet<i16>) -> Self {
        let list: Vec<i16> = all.iter().copied().collect();
        let mut index = HashMap::with_capacity(list.len());
        for (i, lbl) in list.iter().enumerate() {
            index.insert(*lbl, i);
        }
        Self { list, index }
    }
    fn len(&self) -> usize { self.list.len() }
    fn idx(&self, l: i16) -> Option<usize> { self.index.get(&l).copied() }
    fn code_by_idx(&self, idx: usize) -> i16 { self.list[idx] }
}

#[derive(Clone)]
struct PerSigTransfer {
    // Default contributions (None label): vector of (target_sig, mask)
    def_total: Vec<(usize, Weight)>,
    // For concrete label indices: map label_idx -> vector (target_sig, mask)
    per_label: HashMap<usize, Vec<(usize, Weight)>>,
    // Labels that need explicit blocking of default even if contributions end up empty.
    block_labels: BTreeSet<usize>,
}
struct Transfers {
    labels: Labels,
    per_sig: Vec<PerSigTransfer>,
}

fn merge_pairs_by_sig(pairs: impl Iterator<Item=(usize, Weight)>) -> Vec<(usize, Weight)> {
    let mut tmp: HashMap<usize, Weight> = HashMap::new();
    for (t, w) in pairs {
        if !w.is_empty() {
            *tmp.entry(t).or_default() |= &w;
        }
    }
    let mut v: Vec<(usize, Weight)> = tmp.into_iter().collect();
    v.sort_by_key(|(t, _)| *t);
    v
}

fn build_transfers(sigs: &[MacroSig], compiled_steps: &[CompiledStep]) -> Transfers {
    // Collect all labels that appear anywhere (exceptions or explicit labeled ex)
    let mut label_universe: BTreeSet<i16> = BTreeSet::new();
    for sig in sigs {
        for (lbl, _) in &sig.ex {
            label_universe.insert(*lbl);
        }
        for d in &sig.def {
            for &e in &d.exceptions {
                label_universe.insert(e);
            }
        }
    }
    let labels = Labels::new(label_universe);
    let mut per_sig: Vec<PerSigTransfer> = Vec::with_capacity(sigs.len());

    for (sid, sig) in sigs.iter().enumerate() {
        // def_total: union across all def steps
        let def_total = merge_pairs_by_sig(sig.def.iter().flat_map(|d| {
            compiled_steps[d.step_id].by_sig.iter().map(|(t, w)| (*t, w.clone()))
        }));

        // block_labels: ex labels ∪ default exception labels
        let mut block_labels: BTreeSet<usize> = BTreeSet::new();
        for (&lbl, _) in &sig.ex {
            if let Some(li) = labels.idx(lbl) {
                block_labels.insert(li);
            }
        }
        for d in &sig.def {
            for &e in &d.exceptions {
                if let Some(li) = labels.idx(e) {
                    block_labels.insert(li);
                }
            }
        }

        // per_label:
        let mut per_label: HashMap<usize, Vec<(usize, Weight)>> = HashMap::new();

        // Build for each label present in label universe
        for li in 0..labels.len() {
            let lbl_code = labels.code_by_idx(li);

            // Default allowed on lbl?
            let default_blocked_by_ex = sig.ex.contains_key(&lbl_code);
            // If default is allowed, keep only those def steps that don't list this label as exception.
            if !default_blocked_by_ex {
                let default_contrib = merge_pairs_by_sig(sig.def.iter().filter(|d| !d.exceptions.contains(&lbl_code))
                    .flat_map(|d| compiled_steps[d.step_id].by_sig.iter().map(|(t, w)| (*t, w.clone()))));
                if !default_contrib.is_empty() {
                    per_label.entry(li).or_default().extend(default_contrib);
                }
            }
            // Add explicit exception (labeled) transitions
            if let Some(ex_steps) = sig.ex.get(&lbl_code) {
                let ex_contrib = merge_pairs_by_sig(ex_steps.iter()
                    .flat_map(|step_id| compiled_steps[*step_id].by_sig.iter().map(|(t, w)| (*t, w.clone()))));
                if !ex_contrib.is_empty() {
                    per_label.entry(li).or_default().extend(ex_contrib);
                }
            }
            // Normalize merged entries for this label (combine duplicates if default+ex inserted same target twice)
            if let Some(v) = per_label.get_mut(&li) {
                // Merge duplicates produced by separate insertions
                let merged = merge_pairs_by_sig(v.drain(..));
                *v = merged;
            }
        }

        per_sig.push(PerSigTransfer {
            def_total,
            per_label,
            block_labels,
        });
    }

    Transfers { labels, per_sig }
}

// Cached aggregated transitions for a given node gates.
#[derive(Clone)]
struct NodeAggCache {
    // by_label[0] => default (None); by_label[li+1] => labels.list[li]
    by_label: Vec<Vec<(usize, Weight)>>,
    // labels_to_consider as indices into labels (not including default); sorted, deduped.
    labels_to_consider: Vec<usize>,
}

// Aggregate for arbitrary gates using precomputed Transfers.
fn compute_agg_for_gates(
    gates: &HashMap<usize, Weight>,
    transfers: &Transfers,
) -> NodeAggCache {
    let labels_len = transfers.labels.len();
    let mut by_label: Vec<Vec<(usize, Weight)>> = vec![Vec::new(); labels_len + 1];

    // Compute labels to consider: union of labels that either have contributions or require blocking.
    let mut consider: BTreeSet<usize> = BTreeSet::new();

    // Default aggregation
    {
        let mut acc: HashMap<usize, Weight> = HashMap::new();
        for (sig_id, gate_w) in gates {
            if gate_w.is_empty() { continue; }
            let trs = &transfers.per_sig[*sig_id];
            accumulate(&mut acc, &trs.def_total, gate_w);
            // Block-label flags also imply consideration if default would be blocked on that label.
            consider.extend(trs.block_labels.iter().copied());
        }
        if !acc.is_empty() {
            let mut v: Vec<(usize, Weight)> = acc.into_iter().collect();
            v.sort_by_key(|(t, _)| *t);
            by_label[0] = v;
        }
    }

    // Non-default labels aggregation
    for li in 0..labels_len {
        let mut acc: HashMap<usize, Weight> = HashMap::new();
        let mut any_contrib = false;
        for (sig_id, gate_w) in gates {
            if gate_w.is_empty() { continue; }
            let trs = &transfers.per_sig[*sig_id];
            if let Some(pairs) = trs.per_label.get(&li) {
                any_contrib = true;
                accumulate(&mut acc, pairs, gate_w);
            }
            // block label also to be considered even if acc remains empty
            if trs.block_labels.contains(&li) {
                consider.insert(li);
            }
        }
        if any_contrib && !acc.is_empty() {
            let mut v: Vec<(usize, Weight)> = acc.into_iter().collect();
            v.sort_by_key(|(t, _)| *t);
            by_label[li + 1] = v;
            consider.insert(li); // definitely needed
        }
    }

    NodeAggCache {
        by_label,
        labels_to_consider: consider.into_iter().collect(),
    }
}

fn union_weights(vals: impl Iterator<Item=Weight>) -> Weight {
    let mut out = Weight::zeros();
    for v in vals {
        out |= &v;
    }
    out
}

// Equality of two aggregated pairs restricted to intersection I.
// Inputs are sorted by target_sig. Entries with zero mask after &I are ignored.
fn equal_restricted_pairs(a: &[(usize, Weight)], b: &[(usize, Weight)], i: &Weight) -> bool {
    let mut ia = 0usize;
    let mut ib = 0usize;

    loop {
        // advance to next non-zero after masking
        let mut va = None;
        while ia < a.len() {
            let w = &a[ia].1 & i;
            if !w.is_empty() { va = Some((a[ia].0, w)); break; }
            ia += 1;
        }
        let mut vb = None;
        while ib < b.len() {
            let w = &b[ib].1 & i;
            if !w.is_empty() { vb = Some((b[ib].0, w)); break; }
            ib += 1;
        }
        match (va, vb) {
            (None, None) => return true,
            (Some(_), None) | (None, Some(_)) => return false,
            (Some((ta, wa)), Some((tb, wb))) => {
                if ta != tb { return false; }
                if wa != wb { return false; }
                ia += 1;
                ib += 1;
            }
        }
    }
}

// Robust fingerprints for behavior indexing.
// We use masks' internal fp (already robust for large bitsets) and mix label indices and targets.
// This is a conservative prefilter: we still perform exact equality checks after selecting candidates.

fn fp_pairs_with_salt(pairs: &[(usize, Weight)], salt: u64) -> u64 {
    pairs.iter().fold(FP_ZERO, |acc, (t, w)| {
        // Mix target id and mask fp with salt
        let base = mix3((*t as u64).wrapping_mul(FP_K1), w.fp.wrapping_mul(FP_K2), salt);
        mix3(acc, base, FP_K1)
    })
}

fn behavior_fingerprint(cache: &NodeAggCache) -> u64 {
    // Fold default then all labeled entries with their label index.
    let mut fp = FP_ZERO;
    // default (label index 0)
    fp = mix3(fp, fp_pairs_with_salt(&cache.by_label[0], 0), 0xD1E57A_u64);
    for (i, pairs) in cache.by_label.iter().enumerate().skip(1) {
        if pairs.is_empty() { continue; }
        let salt = (i as u64).wrapping_mul(0x9E3779B185EBCA87);
        fp = mix3(fp, fp_pairs_with_salt(pairs, salt), 0xC6A4A7935BD1E995);
    }
    // labels_to_consider do not affect equality; they only force materialization of zero-weight "block default".
    fp
}

fn default_fingerprint(cache: &NodeAggCache) -> u64 {
    fp_pairs_with_salt(&cache.by_label[0], 0xABCD1234_5678_9ABC)
}

// Candidate index with two levels: full-behavior and default-only.
// Safety: Index is used only as a prefilter; we recalc node cache/fps lazily and verify equality afterwards.
struct CandidateIndex {
    by_behavior: HashMap<u64, Vec<usize>>,
    by_default: HashMap<u64, Vec<usize>>,
}
impl CandidateIndex {
    fn new() -> Self {
        Self { by_behavior: HashMap::new(), by_default: HashMap::new() }
    }

    fn ensure_node_indexed(nodes: &mut [CompositionNode], idx: usize) -> (u64, u64) {
        if nodes[idx].cache_dirty || nodes[idx].cached_by_label.is_none() {
            // Recompute cache
            // Note: caller must ensure cache exists before calling this; here we just assert.
            // We keep this function simple; indexing methods will trigger recomputation as needed in their flows.
        }
        let cache = nodes[idx].cached_by_label.as_ref().expect("cache must be present");
        // Recompute fingerprints if missing or cache was dirtied earlier.
        let bfp = if let Some(fp) = nodes[idx].behavior_fp { fp } else {
            let fp = behavior_fingerprint(cache);
            nodes[idx].behavior_fp = Some(fp);
            fp
        };
        let dfp = if let Some(fp) = nodes[idx].default_fp { fp } else {
            let fp = default_fingerprint(cache);
            nodes[idx].default_fp = Some(fp);
            fp
        };
        (bfp, dfp)
    }

    fn index_node(&mut self, nodes: &mut [CompositionNode], idx: usize) {
        let (bfp, dfp) = Self::ensure_node_indexed(nodes, idx);
        self.by_behavior.entry(bfp).or_default().push(idx);
        self.by_default.entry(dfp).or_default().push(idx);
    }

    fn candidates_for_cache<'a>(&'a mut self, nodes: &mut [CompositionNode], cache: &NodeAggCache) -> Vec<usize> {
        let bfp = behavior_fingerprint(cache);
        if let Some(list) = self.by_behavior.get(&bfp) {
            // Filter out stale entries if any
            let mut out = Vec::with_capacity(list.len());
            for &idx in list.iter() {
                // Ensure node has a cache to compute fingerprints.
                if nodes[idx].cached_by_label.is_none() || nodes[idx].cache_dirty {
                    // Cannot verify until cache is recomputed by caller; include now, let caller recompute.
                    out.push(idx);
                } else {
                    let (cur_bfp, _) = Self::ensure_node_indexed(nodes, idx);
                    if cur_bfp == bfp {
                        out.push(idx);
                    }
                }
            }
            if !out.is_empty() {
                return out;
            }
        }
        // Fallback: look by default-only fingerprint
        let dfp = default_fingerprint(cache);
        if let Some(list) = self.by_default.get(&dfp) {
            let mut out = Vec::with_capacity(list.len());
            for &idx in list.iter() {
                if nodes[idx].cached_by_label.is_none() || nodes[idx].cache_dirty {
                    out.push(idx);
                } else {
                    let (_, cur_dfp) = Self::ensure_node_indexed(nodes, idx);
                    if cur_dfp == dfp {
                        out.push(idx);
                    }
                }
            }
            return out;
        }
        Vec::new()
    }
}

// Find or create target composition node for a given gates map.
// Uses behavior indexes to avoid scanning all nodes; falls back to full scan if needed.
// Preserves original merging semantics.
fn find_or_create_target_node(
    map: &HashMap<usize, Weight>,
    nodes: &mut Vec<CompositionNode>,
    transfers: &Transfers,
    cand_index: &mut CandidateIndex,
) -> usize {
    // Compute incoming mask for the new map
    let incoming_mask = union_weights(map.values().cloned());

    // Precompute behavior for the new map once
    let new_cache = compute_agg_for_gates(map, transfers);

    let new_keys = {
        let mut v: Vec<_> = map.keys().copied().collect();
        v.sort_unstable();
        v
    };

    let calc_cost = |cand: &CompositionNode| -> (usize, usize) {
        let current_spec = cand.gates.len();
        let mut inc = 0usize;
        for &sid in &new_keys {
            if !cand.gates.contains_key(&sid) {
                inc += 1;
            }
        }
        (inc, current_spec)
    };

    let mut best_idx: Option<usize> = None;
    let mut best_cost: (usize, usize) = (usize::MAX, usize::MAX);

    // First pass: restrict to candidates with the same full-behavior or at least default fingerprint.
    let mut filtered: Vec<usize> = cand_index.candidates_for_cache(nodes, &new_cache);

    // Ensure the candidate caches are available and current, and update index if needed.
    for &idx in filtered.iter() {
        if nodes[idx].cache_dirty || nodes[idx].cached_by_label.is_none() {
            let cache = compute_agg_for_gates(&nodes[idx].gates, transfers);
            nodes[idx].cached_by_label = Some(cache);
            nodes[idx].cache_dirty = false;
            // Update fingerprints and index entries lazily
            cand_index.index_node(nodes, idx);
        }
    }

    // Evaluate candidates found by index
    for &idx in filtered.iter() {
        let cand = &nodes[idx];
        let inter = &cand.incoming_weight_union & &incoming_mask;

        if inter.is_empty() {
            // Disjoint -> always safe to merge; keep the most specific candidate (min cost)
            let cost = calc_cost(cand);
            if cost < best_cost {
                best_cost = cost;
                best_idx = Some(idx);
            }
            continue;
        }

        // Non-disjoint: behaviors must be equal on inter.
        let cc = cand.cached_by_label.as_ref().unwrap();

        // Compare default (label index 0)
        if !equal_restricted_pairs(&cc.by_label[0], &new_cache.by_label[0], &inter) {
            continue;
        }

        // Compare non-default labels in the union of both 'labels_to_consider'
        let mut lbls: BTreeSet<usize> = BTreeSet::new();
        lbls.extend(cc.labels_to_consider.iter().copied());
        lbls.extend(new_cache.labels_to_consider.iter().copied());

        let mut ok = true;
        for li in lbls {
            let cand_pairs = &cc.by_label[li + 1];
            let new_pairs = &new_cache.by_label[li + 1];
            if !equal_restricted_pairs(cand_pairs, new_pairs, &inter) {
                ok = false;
                break;
            }
        }
        if !ok { continue; }

        // They are compatible on the intersection; prefer best cost
        let cost = calc_cost(cand);
        if cost < best_cost {
            best_cost = cost;
            best_idx = Some(idx);
        }
    }

    // If no good candidate from index, fall back to scanning all nodes (original semantics).
    if best_idx.is_none() {
        for (idx, cand) in nodes.iter_mut().enumerate() {
            let inter = &cand.incoming_weight_union & &incoming_mask;

            if inter.is_empty() {
                let cost = calc_cost(cand);
                if cost < best_cost {
                    best_cost = cost;
                    best_idx = Some(idx);
                }
                continue;
            }

            // Ensure candidate cache
            if cand.cache_dirty || cand.cached_by_label.is_none() {
                let cache = compute_agg_for_gates(&cand.gates, transfers);
                cand.cached_by_label = Some(cache);
                cand.cache_dirty = false;
                cand.behavior_fp = None;
                cand.default_fp = None;
            }
            let cc = cand.cached_by_label.as_ref().unwrap();

            // Compare default (label index 0)
            if !equal_restricted_pairs(&cc.by_label[0], &new_cache.by_label[0], &inter) {
                continue;
            }

            // Compare non-default labels in the union of both 'labels_to_consider'
            let mut lbls: BTreeSet<usize> = BTreeSet::new();
            lbls.extend(cc.labels_to_consider.iter().copied());
            lbls.extend(new_cache.labels_to_consider.iter().copied());

            let mut ok = true;
            for li in lbls {
                let cand_pairs = &cc.by_label[li + 1];
                let new_pairs = &new_cache.by_label[li + 1];
                if !equal_restricted_pairs(cand_pairs, new_pairs, &inter) {
                    ok = false;
                    break;
                }
            }
            if !ok { continue; }

            // They are compatible on the intersection; prefer best cost
            let cost = calc_cost(cand);
            if cost < best_cost {
                best_cost = cost;
                best_idx = Some(idx);
            }
        }
    }

    if let Some(idx) = best_idx {
        // Merge into existing node: expand its incoming weight union
        nodes[idx].incoming_weight_union |= &incoming_mask;
        // Cache may remain valid; fingerprints might need refresh if gates change later.
        idx
    } else {
        // Create a new node
        let new_idx = nodes.len();
        nodes.push(CompositionNode {
            final_weight: None,
            default_target_idx: None,
            default_mask: None,
            exception_targets: BTreeMap::new(),
            exception_masks: BTreeMap::new(),
            gates: HashMap::new(),
            incoming_weight_union: incoming_mask.clone(),
            cached_by_label: None,
            cache_dirty: true,
            behavior_fp: None,
            default_fp: None,
        });
        new_idx
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

// Lazy epsilon-closure cache for only those states that appear as "to" in non-epsilon edges.
struct EpsClosureCache<'a> {
    states: &'a NWAStates,
    fut: &'a [Weight],
    cache: Vec<Option<Vec<(NWAStateID, Weight)>>>,
    scratch_w: Vec<Weight>,
    q: VecDeque<NWAStateID>,
    touched: Vec<NWAStateID>,
}
impl<'a> EpsClosureCache<'a> {
    fn new(states: &'a NWAStates, fut: &'a [Weight]) -> Self {
        let n = states.len();
        Self {
            states,
            fut,
            cache: vec![None; n],
            scratch_w: vec![Weight::zeros(); n],
            q: VecDeque::new(),
            touched: Vec::new(),
        }
    }
    fn get(&mut self, s: NWAStateID) -> &Vec<(NWAStateID, Weight)> {
        if self.cache[s].is_none() {
            let v = eps_closure_masked_vec_one(s, self.states, self.fut, &mut self.scratch_w, &mut self.q, &mut self.touched);
            self.cache[s] = Some(v);
        }
        self.cache[s].as_ref().unwrap()
    }
}

fn update_target_node(
    target_idx: usize,
    map: &HashMap<usize, Weight>,
    nodes: &mut Vec<CompositionNode>,
    in_queue: &mut Vec<bool>,
    work: &mut VecDeque<usize>,
) {
    let mut any_change = false;
    for (sig_id, weight) in map {
        let entry = nodes[target_idx].gates.entry(*sig_id).or_default();
        let new_w = &*entry | weight;
        if new_w != *entry {
            *entry = new_w;
            any_change = true;
        }
    }
    if any_change {
        if target_idx >= in_queue.len() {
            in_queue.resize(target_idx + 1, false);
        }
        if !in_queue[target_idx] {
            in_queue[target_idx] = true;
            work.push_back(target_idx);
        }
        nodes[target_idx].cache_dirty = true;
        nodes[target_idx].behavior_fp = None;
        nodes[target_idx].default_fp = None;
    }
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

    // Zero-length (epsilon-only) future weights:
    // Fε[s] = final[s] ∪ ⋃_{ε s->t}(wε ∧ Fε[t]).
    // This is the exact acceptance mask available at s without consuming input.
    fn compute_epsilon_future_weights(&self) -> Vec<Weight> {
        let n = self.states.len();
        let mut fut: Vec<Weight> = vec![Weight::zeros(); n];
        let mut rev: Vec<Vec<(NWAStateID, &Weight)>> = vec![vec![]; n];

        // Reverse graph only along epsilon edges.
        for p in 0..n {
            for &(t, ref w) in &self.states[p].epsilons {
                if t < n {
                    rev[t].push((p, w));
                }
            }
        }

        let mut q: VecDeque<NWAStateID> = VecDeque::new();
        for s in 0..n {
            if let Some(fw) = &self.states[s].final_weight {
                if !fw.is_empty() {
                    fut[s] = fw.clone();
                    q.push_back(s);
                }
            }
        }

        while let Some(v) = q.pop_front() {
            let fv = fut[v].clone();
            if fv.is_empty() { continue; }
            for &(p, w_pv) in &rev[v] {
                let add = &fv & w_pv;
                if add.is_empty() { continue; }
                let old = &fut[p];
                if (&add & old) != add {
                    fut[p] |= &add;
                    q.push_back(p);
                }
            }
        }
        fut
    }

    fn det_fixpoint(&self) -> DWA {
        // Future acceptance mask (full) is still used to gate epsilon closures for pruning.
        let fut = self.compute_future_weights();
        let eps_future = self.compute_epsilon_future_weights();

        let n = self.states.len();
        if n == 0 {
            return DWA::new();
        }

        // Build MacroSig signatures and pool steps with LAZY epsilon-closures.
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

        let mut eps_cache = EpsClosureCache::new(&self.states, &fut);

        for s in 0..n {
            // Final zero-length acceptance uses epsilon-only future weights.
            let final_acc = if eps_future[s].is_empty() { None } else { Some(eps_future[s].clone()) };

            // Compute default steps; preserve per-default exception sets.
            let mut def_steps: Vec<DefSig> = Vec::new();
            for default in &self.states[s].default {
                let NWADefaultTransition { target: to, weight: wdef, exceptions } = default;
                if *to >= n {
                    continue;
                }
                let eps_pairs = eps_cache.get(*to);
                let pairs_def = apply_weight_to_pairs(eps_pairs, wdef);
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
                    let eps_pairs = eps_cache.get(*to);
                    let pairs_ex = apply_weight_to_pairs(eps_pairs, wlbl);
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

        // Compile steps to be grouped by target signature
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

        // Build transfer tables
        let transfers = build_transfers(&sigs, &compiled_steps);

        // Discover composition nodes
        let mut nodes: Vec<CompositionNode> = Vec::new();
        let mut work: VecDeque<usize> = VecDeque::new();
        let mut cand_index = CandidateIndex::new();

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

        // Initialize start node gates
        let mut init_map: HashMap<usize, Weight> = HashMap::new();
        // Start node gate weights are epsilon-closures of start state's successors-to-sigs:
        for (t, w) in eps_cache.get(self.body.start_state).iter() {
            *init_map.entry(state_to_sig_id[*t]).or_default() |= &w.clone();
        }

        let start_idx = 0;
        nodes.push(CompositionNode {
            final_weight: None,
            default_target_idx: None,
            default_mask: None,
            exception_targets: BTreeMap::new(),
            exception_masks: BTreeMap::new(),
            gates: init_map,
            incoming_weight_union: Weight::all(),
            cached_by_label: None,
            cache_dirty: true,
            behavior_fp: None,
            default_fp: None,
        });
        let mut in_queue = vec![false; 1];
        in_queue[start_idx] = true;
        work.push_back(start_idx);

        while let Some(idx) = work.pop_front() {
            in_queue[idx] = false;
            if let Some(p) = &pb_discover {
                p.inc(1);
            }

            // Ensure cache for this node
            if nodes[idx].cache_dirty || nodes[idx].cached_by_label.is_none() {
                let cache = compute_agg_for_gates(&nodes[idx].gates, &transfers);
                nodes[idx].cached_by_label = Some(cache);
                nodes[idx].cache_dirty = false;
                nodes[idx].behavior_fp = None;
                nodes[idx].default_fp = None;
                // Newly computed cache: index this node for future matches
                cand_index.index_node(&mut nodes, idx);
            }

            if is_debug_level_enabled(5) {
                eprintln!("\nProcessing composition node {}: gates: {:?}", idx, nodes[idx].gates);
            }

            // Phase 1: Read from nodes[idx] and prepare updates.
            let mut default_update_info = None;
            let mut exception_updates_info = BTreeMap::new();

            {
                let node_cache = nodes[idx].cached_by_label.as_ref().unwrap();

                // Default
                let def_pairs = &node_cache.by_label[0];
                let def_total = union_weights(def_pairs.iter().map(|(_, w)| w.clone()));
                if !def_total.is_empty() {
                    let mut map: HashMap<usize, Weight> = HashMap::new();
                    for (ts, w) in def_pairs {
                        *map.entry(*ts).or_default() |= w;
                    }
                    default_update_info = Some((map, def_total));
                }

                // Labels
                for li in &node_cache.labels_to_consider {
                    let pairs = &node_cache.by_label[*li + 1];
                    let total = union_weights(pairs.iter().map(|(_, w)| w.clone()));
                    let lbl_code = transfers.labels.code_by_idx(*li);
                    if total.is_empty() {
                        // Block default
                        exception_updates_info.insert(lbl_code, (None, total));
                    } else {
                        let mut map: HashMap<usize, Weight> = HashMap::new();
                        for (ts, w) in pairs {
                            *map.entry(*ts).or_default() |= w;
                        }
                        exception_updates_info.insert(lbl_code, (Some(map), total));
                    }
                }
            }

            // Phase 2: Mutate nodes.
            let mut resolved_default: Option<(usize, Weight)> = None;
            if let Some((map, def_total)) = default_update_info {
                let target_idx = find_or_create_target_node(&map, &mut nodes, &transfers, &mut cand_index);
                update_target_node(target_idx, &map, &mut nodes, &mut in_queue, &mut work);
                resolved_default = Some((target_idx, def_total));
            }

            let mut resolved_exceptions: BTreeMap<i16, (usize, Weight)> = BTreeMap::new();
            for (lbl_code, (map_opt, total)) in exception_updates_info {
                if let Some(map) = map_opt {
                    let target_idx = find_or_create_target_node(&map, &mut nodes, &transfers, &mut cand_index);
                    update_target_node(target_idx, &map, &mut nodes, &mut in_queue, &mut work);
                    resolved_exceptions.insert(lbl_code, (target_idx, total));
                } else {
                    // Block default
                    resolved_exceptions.insert(lbl_code, (idx, total)); // total is empty here
                }
            }

            // Attach transitions to node
            {
                let node = &mut nodes[idx];
                if let Some((target, mask)) = resolved_default.take() {
                    node.default_target_idx = Some(target);
                    node.default_mask = Some(mask);
                }
                for (lbl, (target_idx, mask)) in resolved_exceptions {
                    node.exception_targets.insert(lbl, target_idx);
                    node.exception_masks.insert(lbl, mask);
                }

                // Final weight accumulation identical to previous approach
                node.final_weight = Into::into(node.gates.iter().fold(Weight::zeros(), |mut acc, (sig_id, gate)| {
                    if let Some(fw) = &sigs[*sig_id].final_w {
                        acc |= &(gate & fw);
                    }
                    acc
                }));
            }

            if let Some(p) = &pb_discover {
                p.set_length(nodes.len() as u64);
            }
        }
        if let Some(p) = pb_discover {
            p.finish_with_message(format!("Discovered {} compositions", nodes.len()));
        }

        // Materialize DWA
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
