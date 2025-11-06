// src/precompute4/weighted_automata/determinization.rs
//
// A radically simplified and aggressively cached determinization.
// Core idea:
//   - Eliminate epsilons on-the-fly with memoized epsilon-closures (OR/AND semiring).
//   - Precompute per-NWA-state "macro steps":
//       step(s, ch) = ε-closure(s) --[ch or default]--> ε-closure(target)
//     as a sparse weighted vector (Vec<(state, weight)>), cached per state and label.
//   - Determinize with "weighted subsets": a D-state is a sparse vector
//     D = Vec<(nwa_state, gate_weight)>. Gate weights come from the path so far;
//     we do NOT store them in DWA state_weight (we keep it empty and carry weights
//     only in edges and final weights), matching how eval_word_weight accumulates.
//   - Next-state computation is a k-way merge across cached step-vectors, each
//     intersected with the current gate. No hashing inside the inner loop, no map
//     updates per-step; just a binary-heap k-way merge over sorted target IDs.
//
// Algorithmic properties:
//   - Correct under the OR/AND idempotent semiring used by Weight.
//   - Every epsilon-star is computed as a monotone fixpoint (worklist), memoized per state.
//   - For each NWA state s, step(s, ch) is computed once and reused everywhere.
//   - Determinization explores only reachable weighted-subsets from the start; each
//     weighted-subset is canonical (sorted by NWA state, normalized by merging same-state gates).
//
// Practical performance advantages:
//   - Eliminates all "macro signature" and "compiled-by-signature" overhead.
//   - Inner loops are cache-friendly and allocation-light (Arc<Vec<...>> reused).
//   - Heavy-weight hashing is only used for interning epsilon-closures, macro-steps,
//     and visited weighted-subsets; the hot path is pure k-way merge.
//
// Notes:
//   - We do not gate epsilon-closures by future weights; while that can prune,
//     its fixpoint can be expensive and brittle. This version focuses on raw throughput.
//   - We keep DWA.state_weight empty; all gating is via edge weights and final weights,
//     precisely mirroring eval_word_weight semantics.
//
// Big-O:
//   - Let Eeps be number of ε edges; E be number of non-ε edges. Each epsilon-closure
//     is computed once per state using a worklist, processing each ε edge only when
//     the propagated weight adds bits (idempotent, monotonically increasing); in
//     practice near-linear.
//   - Each macro step step(s, ch) is built once via two-stage merge over closures.
//   - Determinization visits as many states as required by the language (cannot be
//     avoided in the worst case), but each transition uses a k-way merge over precomputed,
//     immutable step-vectors with trivial per-element work.

#![allow(dead_code)]
#![allow(clippy::needless_borrow)]

use super::common::Weight;
use super::dwa::DWA;
use super::nwa::{NWAStates, NWA};
use crate::precompute4::weighted_automata::NWAStateID;
use crate::profiler::PROGRESS_BAR_ENABLED;
use indicatif::{ProgressBar, ProgressStyle};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::Instant;

// A small helper for mixing 64-bit fingerprints (avoid re-adding dependency on bitset::mix3)
#[inline]
fn mix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

#[inline]
fn mix2(a: u64, b: u64) -> u64 {
    mix64(a ^ (b.rotate_left(13)))
}

// Epsilon-closure cache and macro-step cache
struct Pre {
    states: NWAStates,

    // ε-closure: eps[s] = Arc<Vec<(t, w_eps(s->*t))>>, sorted by t, unique.
    eps: Vec<Option<Arc<Vec<(NWAStateID, Weight)>>>>,

    // final macro: F[s] = ⋃_{t in eps(s)} (w_eps(s->*t) & final[t])
    final_macro: Vec<Option<Weight>>,

    // step caches:
    // default step: step_def[s] = Arc<Vec<(t, w)>>
    step_def: Vec<Option<Arc<Vec<(NWAStateID, Weight)>>>>,
    // exception labeled steps: step_ex[s][lbl] = Arc<Vec<(t, w)>>
    step_ex: Vec<HashMap<i16, Arc<Vec<(NWAStateID, Weight)>>>>,

    // cache for label keys per state (sorted)
    label_keys: Vec<Option<Arc<Vec<i16>>>>,
}

impl Pre {
    fn new(states: NWAStates) -> Self {
        let n = states.len();
        Pre {
            states,
            eps: vec![None; n],
            final_macro: vec![None; n],
            step_def: vec![None; n],
            step_ex: vec![HashMap::new(); n],
            label_keys: vec![None; n],
        }
    }

    #[inline]
    fn n(&self) -> usize {
        self.states.len()
    }

    // Compute ε-closure for a single source s; memoized.
    fn eps_closure(&mut self, s: NWAStateID) -> Arc<Vec<(NWAStateID, Weight)>> {
        if let Some(ref arc) = self.eps[s] {
            return arc.clone();
        }
        let n = self.n();
        let mut map: HashMap<NWAStateID, Weight> = HashMap::new();
        let mut q: VecDeque<NWAStateID> = VecDeque::new();

        // Identity: s can reach itself via empty ε-path with ALL weight.
        map.insert(s, Weight::all());
        q.push_back(s);

        while let Some(u) = q.pop_front() {
            let w_u = map.get(&u).cloned().unwrap_or_else(Weight::zeros);
            if w_u.is_empty() {
                continue;
            }
            // Propagate along ε-edges
            for &(v, ref w_uv) in &self.states[u].epsilons {
                let w_prop = &w_u & w_uv;
                if w_prop.is_empty() {
                    continue;
                }
                match map.get_mut(&v) {
                    Some(old) => {
                        let joined = &*old | &w_prop;
                        if &joined != old {
                            *old = joined;
                            q.push_back(v);
                        }
                    }
                    None => {
                        map.insert(v, w_prop);
                        q.push_back(v);
                    }
                }
            }
        }

        // Normalize to sorted vec by target id
        let mut vec_pairs: Vec<(NWAStateID, Weight)> = map.into_iter().collect();
        vec_pairs.sort_by_key(|(t, _)| *t);

        let arc = Arc::new(vec_pairs);
        self.eps[s] = Some(arc.clone());
        arc
    }

    // Compute final macro F[s] once: OR over eps targets of (w_eps & final_weight[target])
    fn final_macro_of(&mut self, s: NWAStateID) -> Option<Weight> {
        if let Some(ref w) = self.final_macro[s] {
            return Some(w.clone());
        }
        let eps = self.eps_closure(s);
        let mut acc: Option<Weight> = None;
        for (t, w_eps) in eps.iter() {
            if let Some(fw) = &self.states[*t].final_weight {
                let c = w_eps & fw;
                if !c.is_empty() {
                    if let Some(a) = &mut acc {
                        *a |= &c;
                    } else {
                        acc = Some(c);
                    }
                }
            }
        }
        self.final_macro[s] = acc.clone();
        acc
    }

    // All explicit labels (keys) for state s; sorted
    fn labels_of(&mut self, s: NWAStateID) -> Arc<Vec<i16>> {
        if let Some(ref arc) = self.label_keys[s] {
            return arc.clone();
        }
        let ks: Vec<i16> = self.states[s].transitions.keys().copied().collect();
        let arc = Arc::new(ks);
        self.label_keys[s] = Some(arc.clone());
        arc
    }

    // Macro step for (s, label):
    // For a given s and label (Some(lbl) or None for default macro),
    // we compute contributions by:
    //   For each (u, wsu) in eps(s):
    //     let trans(u, label) = explicit on label, else default, else None -> skip
    //     let (v, w_uv) = chosen transition
    //     w_pre = wsu & w_uv
    //     For each (t, w_vt) in eps(v):
    //         contribute t with weight (w_pre & w_vt)
    // Result is sorted by t; merged by OR for duplicate t.
    fn step_for_label(
        &mut self,
        s: NWAStateID,
        label: Option<i16>,
    ) -> Arc<Vec<(NWAStateID, Weight)>> {
        // Cache lookup
        match label {
            None => {
                if let Some(ref arc) = self.step_def[s] {
                    return arc.clone();
                }
            }
            Some(lbl) => {
                if let Some(arc) = self.step_ex[s].get(&lbl) {
                    return arc.clone();
                }
            }
        }

        let eps_s = self.eps_closure(s);
        let mut out: HashMap<NWAStateID, Weight> = HashMap::new();

        for (u, w_su) in eps_s.iter() {
            // Choose transition according to label or default
            let step_opt = match label {
                Some(lbl) => self.states[*u].get_transition(lbl),
                None => self.states[*u].default.as_ref(),
            };
            if let Some((v, w_edge)) = step_opt {
                let w_pre = w_su & w_edge;
                if w_pre.is_empty() {
                    continue;
                }
                let eps_v = self.eps_closure(*v);
                for (t, w_vt) in eps_v.iter() {
                    let w = &w_pre & w_vt;
                    if w.is_empty() {
                        continue;
                    }
                    if let Some(old) = out.get_mut(t) {
                        *old |= &w;
                    } else {
                        out.insert(*t, w);
                    }
                }
            }
        }

        let mut v: Vec<(NWAStateID, Weight)> = out.into_iter().collect();
        v.sort_by_key(|(t, _)| *t);
        let arc = Arc::new(v);

        match label {
            None => {
                self.step_def[s] = Some(arc.clone());
            }
            Some(lbl) => {
                self.step_ex[s].insert(lbl, arc.clone());
            }
        }
        arc
    }
}

// A determinized state is a sparse vector of (nwa_state, gate_weight) pairs.
// Canonical, sorted by nwa_state; duplicate keys merged by OR.
// We intern them via HashMap using a 64-bit fingerprint.
#[derive(Clone)]
struct WSubsetKey {
    entries: Arc<Vec<(NWAStateID, Weight)>>,
    fp: u64,
}
impl PartialEq for WSubsetKey {
    fn eq(&self, other: &Self) -> bool {
        if self.fp != other.fp {
            return false;
        }
        if Arc::ptr_eq(&self.entries, &other.entries) {
            return true;
        }
        if self.entries.len() != other.entries.len() {
            return false;
        }
        for (a, b) in self.entries.iter().zip(other.entries.iter()) {
            if a.0 != b.0 || a.1 != b.1 {
                return false;
            }
        }
        true
    }
}
impl Eq for WSubsetKey {}
impl Hash for WSubsetKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.fp);
    }
}
impl WSubsetKey {
    fn new(mut items: Vec<(NWAStateID, Weight)>) -> Self {
        // Normalize: merge dup keys by OR, drop empties, sort by state id.
        items.sort_by_key(|(sid, _)| *sid);
        let mut norm: Vec<(NWAStateID, Weight)> = Vec::with_capacity(items.len());
        for (sid, w) in items.into_iter() {
            if w.is_empty() {
                continue;
            }
            if let Some((last_sid, ref mut last_w)) = norm.last_mut() {
                if *last_sid == sid {
                    *last_w |= &w;
                    continue;
                }
            }
            norm.push((sid, w));
        }
        let arc = Arc::new(norm);
        let mut fp = 0u64;
        for (sid, w) in arc.iter() {
            fp = mix2(fp, (*sid as u64));
            fp = mix2(fp, w.fp);
        }
        WSubsetKey { entries: arc, fp }
    }

    fn from_arc(arc: Arc<Vec<(NWAStateID, Weight)>>) -> Self {
        // Assumes input is already normalized and sorted!
        let mut fp = 0u64;
        for (sid, w) in arc.iter() {
            fp = mix2(fp, (*sid as u64));
            fp = mix2(fp, w.fp);
        }
        WSubsetKey { entries: arc, fp }
    }
}

// Utility: OR-sum of weights in a vector of (sid, weight)
#[inline]
fn union_all_weights(v: &[(NWAStateID, Weight)]) -> Option<Weight> {
    let mut acc: Option<Weight> = None;
    for (_, w) in v {
        if let Some(a) = &mut acc {
            *a |= w;
        } else {
            acc = Some(w.clone());
        }
    }
    acc
}

// Merge k sorted step-vectors with per-vector gate weights.
// Each input is an Arc<Vec<(target, weight)>> and a &Weight gate G.
// The output is the sorted union over targets with weight = OR over (G & weight_in_that_vector).
fn merge_stepvecs_with_gates(
    inputs: &[(Arc<Vec<(NWAStateID, Weight)>>, &Weight)],
) -> Arc<Vec<(NWAStateID, Weight)>> {
    // Fast exits
    if inputs.is_empty() {
        return Arc::new(Vec::new());
    }
    if inputs.len() == 1 {
        // Apply single gate and filter empties
        let (sv, gate) = &inputs[0];
        if gate.is_all_fast() {
            return sv.clone();
        }
        let mut out: Vec<(NWAStateID, Weight)> = Vec::with_capacity(sv.len());
        for (t, w) in sv.iter() {
            let c = *gate & w;
            if !c.is_empty() {
                out.push((*t, c));
            }
        }
        return Arc::new(out);
    }

    // Maintain a min-heap keyed by current target id; each heap entry points to (list_idx, pos)
    #[derive(Copy, Clone)]
    struct Cursor {
        list_idx: usize,
        pos: usize,
        target: NWAStateID,
    }
    impl PartialEq for Cursor {
        fn eq(&self, other: &Self) -> bool {
            self.target == other.target && self.list_idx == other.list_idx && self.pos == other.pos
        }
    }
    impl Eq for Cursor {}
    impl PartialOrd for Cursor {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Cursor {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            // We want min-heap, so reverse
            self.target.cmp(&other.target).reverse().then(self.list_idx.cmp(&other.list_idx).reverse())
        }
    }

    let mut heap: BinaryHeap<Cursor> = BinaryHeap::new();
    let mut lists: Vec<(&[ (NWAStateID, Weight) ], &Weight)> = Vec::with_capacity(inputs.len());
    for (idx, (arc, gate)) in inputs.iter().enumerate() {
        if arc.is_empty() {
            continue;
        }
        lists.push((arc.as_slice(), *gate));
        let t0 = arc[0].0;
        heap.push(Cursor { list_idx: idx, pos: 0, target: t0 });
    }
    if heap.is_empty() {
        return Arc::new(Vec::new());
    }

    let mut out: Vec<(NWAStateID, Weight)> = Vec::new();

    while let Some(Cursor { list_idx, pos, target }) = heap.pop() {
        // Gather all contributions for this target
        let mut acc: Option<Weight> = None;

        // First contribution
        let (list_slice, gate) = &lists[list_idx];
        let (t, w) = &list_slice[pos];
        debug_assert_eq!(*t, target);
        let c = *gate & w;
        if !c.is_empty() {
            acc = Some(c);
        }

        // Pull more from other lists with same target at top
        // We need to peek others; easiest: loop while heap.top has same target
        let mut popped: Vec<Cursor> = Vec::new();
        while let Some(top) = heap.peek() {
            if top.target != target {
                break;
            }
            popped.push(heap.pop().unwrap());
        }
        for cur in popped.into_iter() {
            let (slice, gate2) = &lists[cur.list_idx];
            let (_, w2) = &slice[cur.pos];
            let c2 = *gate2 & w2;
            if !c2.is_empty() {
                if let Some(a) = &mut acc {
                    *a |= &c2;
                } else {
                    acc = Some(c2);
                }
            }
            // advance this cursor
            let next_pos = cur.pos + 1;
            if next_pos < slice.len() {
                let next_target = slice[next_pos].0;
                heap.push(Cursor { list_idx: cur.list_idx, pos: next_pos, target: next_target });
            }
        }

        // Advance the original cursor
        let next_pos0 = pos + 1;
        if next_pos0 < lists[list_idx].0.len() {
            let next_target0 = lists[list_idx].0[next_pos0].0;
            heap.push(Cursor { list_idx, pos: next_pos0, target: next_target0 });
        }

        if let Some(wsum) = acc {
            out.push((target, wsum));
        }
    }

    Arc::new(out)
}

// Compute union of explicit labels across a weighted subset (no dedup duplicates in loop, do a final dedup)
fn gather_exception_labels(pre: &mut Pre, subset: &[(NWAStateID, Weight)]) -> Vec<i16> {
    let mut labels: Vec<i16> = Vec::new();
    for (s, _) in subset.iter() {
        let ks = pre.labels_of(*s);
        labels.extend_from_slice(&ks);
    }
    if labels.is_empty() {
        return labels;
    }
    labels.sort_unstable();
    labels.dedup();
    labels
}

impl NWA {
    /// New determinization: on-the-fly ε-elimination, k-way merge of cached macro steps,
    /// weighted-subset construction. No "macro signatures", no "compiled-by-sig" indirections.
    pub fn determinize_to_dwa(&self) -> DWA {
        let now = Instant::now();
        let mut tmp = self.clone();
        // Keep simplification light but useful to shrink graph (unreachables, SCC collapse of ALL-ε)
        tmp.simplify();

        let mut pre = Pre::new(tmp.states.clone());
        let start = tmp.body.start_state;

        // Initial weighted subset = ε-closure(start)
        let init_eps = pre.eps_closure(start);
        let init_subset = WSubsetKey::from_arc(init_eps);

        // Prepare DWA
        let mut dwa = DWA::new();
        dwa.states.0.clear();
        let start_id = dwa.states.add_state();
        dwa.body.start_state = start_id;

        // Visited map: weighted-subset key -> DWA state id
        let mut visited: HashMap<WSubsetKey, usize> = HashMap::new();
        visited.insert(init_subset.clone(), start_id);

        // Worklist of subset keys to process
        let mut q: VecDeque<WSubsetKey> = VecDeque::new();
        q.push_back(init_subset.clone());

        // Optional progress
        let pb = if PROGRESS_BAR_ENABLED {
            let p = ProgressBar::new(1);
            p.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [Determinizing (eps+merge): {elapsed_precise}] \
                               [{wide_bar:.cyan/blue}] {pos}/{len} states ({percent}%, {eta})")
                    .expect("progress-bar"),
            );
            p.set_position(0);
            p.set_length(1);
            Some(p)
        } else {
            None
        };

        let mut processed = 0usize;

        while let Some(subkey) = q.pop_front() {
            processed += 1;
            if let Some(ref p) = pb {
                p.set_position(processed as u64);
                p.set_length(visited.len() as u64);
            }

            let d_id = *visited.get(&subkey).expect("subset must be visited");
            let subset: &[(NWAStateID, Weight)] = &subkey.entries;

            // Final weight: OR over (gate & final_macro[s])
            let mut final_acc: Option<Weight> = None;
            for (s, gate) in subset.iter() {
                if let Some(fw) = pre.final_macro_of(*s) {
                    let c = gate & &fw;
                    if !c.is_empty() {
                        if let Some(a) = &mut final_acc {
                            *a |= &c;
                        } else {
                            final_acc = Some(c);
                        }
                    }
                }
            }
            dwa.states[d_id].final_weight = final_acc;

            // Build default transition by merging all default macro steps.
            // Each source contributes step_def[s] if available, gated by its gate.
            let mut def_inputs: Vec<(Arc<Vec<(NWAStateID, Weight)>>, &Weight)> = Vec::new();
            for (s, gate) in subset.iter() {
                if let Some(ref arc) = pre.step_def[*s] {
                    // already computed default step
                    def_inputs.push((arc.clone(), gate));
                } else {
                    // try compute; step_for_label(None) will store into cache, even if empty
                    let arc = pre.step_for_label(*s, None);
                    if !arc.is_empty() {
                        def_inputs.push((arc, gate));
                    }
                }
            }
            let def_target_arc = merge_stepvecs_with_gates(&def_inputs);
            let def_weight = union_all_weights(def_target_arc.as_slice());

            // Prepare set of exception labels (union of explicit labels across sources)
            let labels = gather_exception_labels(&mut pre, subset);

            // For each label, merge step_ex[s][label] where present, falling back to default for others.
            // We'll compare (target subset, edge weight) to default; if identical, skip explicit edge.
            let mut def_target_id_opt: Option<usize> = None;
            if !def_target_arc.is_empty() && def_weight.as_ref().map_or(false, |w| !w.is_empty()) {
                let def_key = WSubsetKey::from_arc(def_target_arc.clone());
                let def_target_id = if let Some(id) = visited.get(&def_key) {
                    *id
                } else {
                    let nid = dwa.states.add_state();
                    visited.insert(def_key.clone(), nid);
                    q.push_back(def_key);
                    nid
                };
                // Safe to unwrap: we checked non-empty and non-zero above
                dwa
                    .set_default_transition(d_id, def_target_id, def_weight.clone().unwrap())
                    .expect("set_default_transition");
                def_target_id_opt = Some(def_target_id);
            }

            // Helper comparator to check whether exception equals default by both target-subset and weight.
            let mut def_key_opt: Option<WSubsetKey> = None;
            if !def_target_arc.is_empty() {
                def_key_opt = Some(WSubsetKey::from_arc(def_target_arc.clone()));
            }

            // Build exceptions
            for lbl in labels {
                let mut ex_inputs: Vec<(Arc<Vec<(NWAStateID, Weight)>>, &Weight)> = Vec::with_capacity(subset.len());
                for (s, gate) in subset.iter() {
                    // If s has explicit, use it; else use default
                    if pre.states[*s].transitions.contains_key(&lbl) {
                        let arc = pre.step_for_label(*s, Some(lbl));
                        if !arc.is_empty() {
                            ex_inputs.push((arc, gate));
                        }
                    } else {
                        // Fallback to default if any
                        let arc = pre.step_for_label(*s, None);
                        if !arc.is_empty() {
                            ex_inputs.push((arc, gate));
                        }
                    }
                }
                if ex_inputs.is_empty() {
                    continue;
                }
                let ex_target_arc = merge_stepvecs_with_gates(&ex_inputs);
                if ex_target_arc.is_empty() {
                    continue;
                }
                let ex_weight = union_all_weights(ex_target_arc.as_slice());
                if ex_weight.as_ref().map_or(true, |w| w.is_empty()) {
                    continue;
                }

                // Compare to default (both structure and edge weight)
                let mut same_as_default = false;
                if let (Some(ref def_k), Some(ref dw)) = (&def_key_opt, &def_weight) {
                    let ex_k = WSubsetKey::from_arc(ex_target_arc.clone());
                    if &ex_k == def_k && ex_weight.as_ref().unwrap() == dw {
                        same_as_default = true;
                    }
                }
                if same_as_default {
                    continue;
                }

                // Create/get target state
                let ex_key = WSubsetKey::from_arc(ex_target_arc.clone());
                let target_id = if let Some(id) = visited.get(&ex_key) {
                    *id
                } else {
                    let nid = dwa.states.add_state();
                    visited.insert(ex_key.clone(), nid);
                    q.push_back(ex_key);
                    nid
                };
                dwa
                    .add_transition(d_id, lbl, target_id, ex_weight.unwrap())
                    .expect("add_transition");
            }
        }

        if let Some(p) = pb {
            p.finish_with_message(format!("Determinized to {} states", visited.len()));
        }

        // A quick simplification pass to normalize edges and prune unreachable (cheap)
        // to minimize final DWA size and canonicalize default/exception structure.
        let mut dwa2 = dwa.clone();
        dwa2.simplify();
        // Return simpler if it reduced states, else return original
        if dwa2.states.len() <= dwa.states.len() {
            crate::debug!(4, "NWA::determinize_to_dwa (new) took: {:?}", now.elapsed());
            dwa2
        } else {
            crate::debug!(4, "NWA::determinize_to_dwa (new) took: {:?}", now.elapsed());
            dwa
        }
    }
}
