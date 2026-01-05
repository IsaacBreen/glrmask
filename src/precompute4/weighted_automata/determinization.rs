#![allow(dead_code)]
#![allow(clippy::needless_borrow)]

use chrono::Local;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rustc_hash::FxHashMap;
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::ops::BitOrAssign;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use crate::precompute4::weighted_automata::test_weighted_automata;
use super::common::{DETERMINIZE_DEBUG, Label, NWAStateID, Weight};
use super::dwa::DWA;
use super::nwa::{NWA, NWAStates};

// Global counter for determinizations
static DETERMINIZE_COUNT: AtomicUsize = AtomicUsize::new(0);
static TOTAL_DETERMINIZE_TIME_MS: AtomicUsize = AtomicUsize::new(0);

pub fn reset_determinize_stats() {
    DETERMINIZE_COUNT.store(0, AtomicOrdering::SeqCst);
    TOTAL_DETERMINIZE_TIME_MS.store(0, AtomicOrdering::SeqCst);
}

pub fn get_determinize_stats() -> (usize, usize) {
    (DETERMINIZE_COUNT.load(AtomicOrdering::SeqCst), TOTAL_DETERMINIZE_TIME_MS.load(AtomicOrdering::SeqCst))
}

// ============================================================================
// Common Types & Helpers
// ============================================================================

// Invariants: strictly sorted by NWAStateID, no duplicate IDs, no empty Weights.
type WeightedSubset = Vec<(NWAStateID, Weight)>;

fn is_zero(w: &Weight) -> bool { w.is_empty() }

/// A pre-hashed wrapper for a weighted subset using sorted Vec for fast iteration.
#[derive(Clone)]
struct HashedSubset {
    inner: Vec<(NWAStateID, Weight)>,  // Sorted by NWAStateID
    hash: u64,
}

impl HashedSubset {
    fn from_btreemap(map: BTreeMap<NWAStateID, Weight>) -> Self {
        use rustc_hash::FxHasher;
        let inner: Vec<_> = map.into_iter().collect();
        let mut hasher = FxHasher::default();
        for (k, v) in &inner {
            k.hash(&mut hasher);
            v.hash(&mut hasher);
        }
        let hash = hasher.finish();
        Self { inner, hash }
    }
    
    fn from_fxhashmap(map: FxHashMap<NWAStateID, Weight>) -> Self {
        use rustc_hash::FxHasher;
        let mut inner: Vec<_> = map.into_iter().collect();
        inner.sort_unstable_by_key(|(k, _)| *k);
        let mut hasher = FxHasher::default();
        for (k, v) in &inner {
            k.hash(&mut hasher);
            v.hash(&mut hasher);
        }
        let hash = hasher.finish();
        Self { inner, hash }
    }

    fn new(inner: BTreeMap<NWAStateID, Weight>) -> Self {
        Self::from_btreemap(inner)
    }
    
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

impl PartialEq for HashedSubset {
    fn eq(&self, other: &Self) -> bool {
        self.hash == other.hash && self.inner == other.inner
    }
}
impl Eq for HashedSubset {}

impl Hash for HashedSubset {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.hash.hash(state);
    }
}

// ============================================================================
// NWA Determinization Interface
// ============================================================================

impl NWA {
    /// The primary entry point for determinization.
    ///
    /// This defaults to the **Simple** strategy for performance.
    pub fn determinize(&self) -> DWA {
        self.determinize_robust()
    }

    /// Determinizes the NWA using a robust strategy with precomputed epsilon closures.
    ///
    /// **Characteristics:**
    /// - Precomputes epsilon reachability to handle complex epsilon graphs.
    /// - Less prone to state explosion in complex topologies.
    /// - Includes a "singleton loop" heuristic optimization.
    /// - Enforces a hard state limit (250,000) to prevent OOM, dumping the NWA if exceeded.
    /// - Displays a progress bar for large automata.
    /// - **Formerly:** `determinize_to_dwa2`
    pub fn determinize_robust(&self) -> DWA {
        // 1. Try Heuristic Optimization
        if let Some(dwa) = try_build_singleton_loop_union(self) {
            return dwa;
        }

        // 2. Setup Limits and Progress
        const STATE_LIMIT: usize = 250_000; 
        if self.states.0.is_empty() {
            return DWA::new();
        }

        crate::debug!(5, "Determinization: Precomputing epsilon closures...");
        
        // 3. Precompute Reachability
        let eps_reach = precompute_all_epsilon_closures(&self.states);

        // 4. Configure Progress Bar
        // We only show the progress bar if determinization takes longer than 100ms.
        let start_time = std::time::Instant::now();
        let mut pbar_data: Option<(MultiProgress, ProgressBar)> = None;

        // 5. Initialize Determinizer
        let mut det = Determinizer::new(self, &eps_reach);

        // 6. Initial State Construction
        let mut start_map: HashMap<NWAStateID, Weight> = HashMap::new();
        for &s in &self.body.start_states {
            if s < eps_reach.len() {
                for (v, w_reach) in &eps_reach[s] {
                    start_map.entry(*v)
                        .and_modify(|acc| *acc |= w_reach)
                        .or_insert_with(|| w_reach.clone());
                }
            }
        }
        
        let mut start_subset: WeightedSubset = start_map.into_iter().collect();
        start_subset.sort_unstable_by(|a, b| a.0.cmp(&b.0));

        let start_id = det.register_closure(start_subset);
        det.dwa.body.start_state = start_id;

        // 7. Main Expansion Loop
        let mut processed_count = 0;
        while let Some(sid) = det.queue.pop_front() {
            // Safety Guard
            if det.seen.len() > STATE_LIMIT {
                let timestamp = Local::now().format("%Y%m%d-%H%M%S");
                let filename = format!("nwa_dump_{}.json", timestamp);
                crate::debug!(5, "Determinization state limit ({}) exceeded. Dumping NWA to {} and panicking.", STATE_LIMIT, filename);
                let f = std::fs::File::create(&filename).expect("Unable to create dump file");
                serde_json::to_writer(f, self).expect("Unable to write NWA to file");
                panic!("Determinization aborted after reaching {} states.", STATE_LIMIT);
            }

            // Progress Update
            // Initialize progress bar if > 100ms has elapsed and it's not yet created
            if pbar_data.is_none() && start_time.elapsed().as_millis() > 100 {
                let mp = MultiProgress::new();
                let pb = mp.add(ProgressBar::new(1));
                pb.set_style(
                    ProgressStyle::default_bar()
                        .template("{spinner:.green} [{elapsed_precise}] States: [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
                        .unwrap()
                        .progress_chars("#>-"),
                );
                pb.set_message("Determinizing NWA");
                pbar_data = Some((mp, pb));
            }

            if let Some((_, pb)) = &pbar_data {
                let total_states = det.seen.len();
                pb.set_length(total_states as u64);
                pb.set_position(processed_count as u64);
                pb.set_message(format!("Expanding state {}/{}", processed_count + 1, total_states));
            }

            det.expand_state(sid);
            processed_count += 1;
        }
        
        if let Some((_, pb)) = pbar_data {
            pb.finish_with_message("Determinization complete");
        }

        // 8. Debug Verification
        if DETERMINIZE_DEBUG {
            let rustfst_dwa = self.determinize_to_dwa_with_rustfst();
            crate::debug!(5, "[DETERMINIZE_DEBUG] Comparing custom determinization with rustfst...");
            test_weighted_automata::stochastic_equivalence_test(det.dwa.clone(), rustfst_dwa);
        }

        det.dwa
    }

    /// Determinizes the NWA using a simple on-the-fly strategy.
    ///
    /// **Characteristics:**
    /// - Performs epsilon closure dynamically during expansion.
    /// - Faster initialization (no precomputation).
    /// - More prone to state explosion if epsilon chains are deep.
    /// - **Formerly:** `_determinize`
    pub fn determinize_simple(&self) -> DWA {
        let call_count = DETERMINIZE_COUNT.fetch_add(1, AtomicOrdering::SeqCst) + 1;
        let det_fn_start = std::time::Instant::now();
        
        let mut dwa = DWA::new();
        dwa.states.0.clear();
        
        // Log NWA stats for large automata
        let num_nwa_states = self.states.len();
        let total_eps: usize = self.states.0.iter().map(|s| s.epsilons.len()).sum();
        if num_nwa_states > 100_000 {
            crate::debug!(5, "NWA determinize_simple #{}: {} states, {} epsilon transitions", call_count, num_nwa_states, total_eps);
        }

        // Use pre-hashed subset map for faster lookups
        let mut subset_map: FxHashMap<HashedSubset, NWAStateID> = FxHashMap::default();
        let mut worklist: VecDeque<HashedSubset> = VecDeque::new();

        // Timing
        let mut time_epsilon_closure = 0u64;
        let mut time_collect_transitions = 0u64;
        let mut time_final_weights = 0u64;
        let mut time_build_edges = 0u64;
        let mut time_map_lookup = 0u64;
        let mut epsilon_closure_calls = 0u64;
        let mut total_subset_size = 0u64;
        let mut max_subset_size = 0usize;
        let det_start = std::time::Instant::now();

        // Initial States
        let mut start_subset = BTreeMap::new();
        for &s in &self.body.start_states {
            if s < self.states.len() {
                start_subset.insert(s, Weight::all());
            }
        }

        let initial_subset = self.epsilon_closure_simple(&start_subset);

        if !initial_subset.is_empty() {
            let start_id = dwa.add_state();
            dwa.body.start_state = start_id;
            let hashed_initial = HashedSubset::new(initial_subset);
            subset_map.insert(hashed_initial.clone(), start_id);
            worklist.push_back(hashed_initial);
        } else {
            let start_id = dwa.add_state();
            dwa.body.start_state = start_id;
        }

        // Expansion Loop
        while let Some(subset) = worklist.pop_front() {
            let from_dwa_id = *subset_map.get(&subset).unwrap();
            total_subset_size += subset.inner.len() as u64;
            max_subset_size = max_subset_size.max(subset.inner.len());

            let t_final = std::time::Instant::now();

            // Compute Final Weights
            let mut final_weight = Weight::zeros();
            for (nwa_id, path_weight) in &subset.inner {
                if let Some(fw) = &self.states[*nwa_id].final_weight {
                    final_weight |= &(path_weight & fw);
                }
            }
            if !final_weight.is_empty() {
                dwa.states[from_dwa_id].final_weight = Some(final_weight);
            }
            time_final_weights += t_final.elapsed().as_nanos() as u64;

            // Collect Transitions
            let t_collect = std::time::Instant::now();
            
            // Use FxHashMap for faster transition collection, then sort at the end
            let mut transitions: FxHashMap<Label, FxHashMap<NWAStateID, Weight>> = FxHashMap::default();
            let mut edge_weights: FxHashMap<Label, Weight> = FxHashMap::default();
            for (nwa_id, path_weight) in &subset.inner {
                for (label, targets) in &self.states[*nwa_id].transitions {
                    for (target_nwa_id, trans_weight) in targets {
                        let next_path_weight = path_weight & trans_weight;
                        if !next_path_weight.is_empty() {
                            edge_weights.entry(*label).or_insert_with(Weight::zeros).bitor_assign(&next_path_weight);
                            let entry = transitions.entry(*label).or_default();
                            entry.entry(*target_nwa_id).or_insert_with(Weight::zeros).bitor_assign(&next_path_weight);
                        }
                    }
                }
            }
            time_collect_transitions += t_collect.elapsed().as_nanos() as u64;

            // Build Edges
            let t_edges = std::time::Instant::now();
            for (label, next_subset_pre_closure) in transitions {
                let t_eps = std::time::Instant::now();
                let next_subset = self.epsilon_closure_simple_fx(&next_subset_pre_closure);
                time_epsilon_closure += t_eps.elapsed().as_nanos() as u64;
                epsilon_closure_calls += 1;
                
                if next_subset.is_empty() {
                    continue;
                }
                
                let w_edge = edge_weights.remove(&label).unwrap();
                let w_edge_inv = !&w_edge;

                // Normalize weights in the subset by dividing by w_edge.
                // Division in Boolean semiring (loosening): w / v = w | !v.
                let normalized_subset: FxHashMap<NWAStateID, Weight> = next_subset
                    .into_iter()
                    .map(|(id, w)| (id, w | &w_edge_inv))
                    .collect();

                let t_map = std::time::Instant::now();
                let hashed_next = HashedSubset::from_fxhashmap(normalized_subset);
                let to_dwa_id = *subset_map.entry(hashed_next.clone()).or_insert_with(|| {
                    let new_id = dwa.add_state();
                    worklist.push_back(hashed_next);
                    new_id
                });
                time_map_lookup += t_map.elapsed().as_nanos() as u64;
                dwa.add_transition(from_dwa_id, label, to_dwa_id, w_edge).unwrap();
            }
            time_build_edges += t_edges.elapsed().as_nanos() as u64;
        }
        
        let total_time = det_start.elapsed().as_millis();
        if total_time > 200 {
            let num_dfa_states = dwa.states.len();
            let avg_subset = if num_dfa_states > 0 { total_subset_size / num_dfa_states as u64 } else { 0 };
            crate::debug!(5, "NWA determinize_simple timing: total={}ms, epsilon_closure={}ms ({} calls), collect_trans={}ms, final_weights={}ms, build_edges={}ms, map_lookup={}ms, avg_subset={}, max_subset={}",
                total_time,
                time_epsilon_closure / 1_000_000,
                epsilon_closure_calls,
                time_collect_transitions / 1_000_000,
                time_final_weights / 1_000_000,
                time_build_edges / 1_000_000,
                time_map_lookup / 1_000_000,
                avg_subset,
                max_subset_size);
        }
        
        // Update global stats
        TOTAL_DETERMINIZE_TIME_MS.fetch_add(det_fn_start.elapsed().as_millis() as usize, AtomicOrdering::SeqCst);
        
        dwa
    }

    // Helper specific to the 'Simple' strategy - FxHashMap version for performance
    fn epsilon_closure_simple_fx(&self, subset: &FxHashMap<NWAStateID, Weight>) -> FxHashMap<NWAStateID, Weight> {
        let mut closure: FxHashMap<NWAStateID, Weight> = subset.clone();
        let mut worklist: VecDeque<NWAStateID> = subset.keys().copied().collect();

        while let Some(u) = worklist.pop_front() {
            let u_weight = closure.get(&u).unwrap().clone();
            if u >= self.states.len() {
                continue;
            }
            for (v, eps_weight) in &self.states[u].epsilons {
                let v_new_weight = &u_weight & eps_weight;
                if !v_new_weight.is_empty() {
                    let v_current_weight = closure.entry(*v).or_insert_with(Weight::zeros);
                    let combined = &*v_current_weight | &v_new_weight;
                    if combined != *v_current_weight {
                        *v_current_weight = combined;
                        worklist.push_back(*v);
                    }
                }
            }
        }
        closure
    }

    // Helper specific to the 'Simple' strategy
    fn epsilon_closure_simple(&self, subset: &BTreeMap<NWAStateID, Weight>) -> BTreeMap<NWAStateID, Weight> {
        let mut closure = subset.clone();
        let mut worklist: VecDeque<NWAStateID> = subset.keys().copied().collect();

        while let Some(u) = worklist.pop_front() {
            let u_weight = closure.get(&u).unwrap().clone();
            if u >= self.states.len() {
                continue;
            }
            for (v, eps_weight) in &self.states[u].epsilons {
                let v_new_weight = &u_weight & eps_weight;
                if !v_new_weight.is_empty() {
                    let v_current_weight = closure.entry(*v).or_insert_with(Weight::zeros);
                    let combined = &*v_current_weight | &v_new_weight;
                    if combined != *v_current_weight {
                        *v_current_weight = combined;
                        worklist.push_back(*v);
                    }
                }
            }
        }
        closure
    }
}

// ============================================================================
// Strategy: Robust / Precomputed Implementation Details
// ============================================================================

struct Determinizer<'a> {
    nwa: &'a NWA,
    eps_reach: &'a [WeightedSubset],
    
    // Map from canonical closure (Sorted Vec) to DWA State ID
    seen: HashMap<WeightedSubset, usize>,
    queue: VecDeque<usize>,
    // Store the closure for each DWA state
    closures: Vec<WeightedSubset>,
    
    dwa: DWA,
}

impl<'a> Determinizer<'a> {
    fn new(nwa: &'a NWA, eps_reach: &'a [WeightedSubset]) -> Self {
        let mut dwa = DWA::new();
        dwa.states.0.clear();
        dwa.body.start_state = 0;
        Determinizer {
            nwa,
            eps_reach,
            seen: HashMap::new(),
            queue: VecDeque::new(),
            closures: Vec::new(),
            dwa,
        }
    }

    fn register_closure(&mut self, closure: WeightedSubset) -> usize {
        if let Some(&id) = self.seen.get(&closure) {
            return id;
        }

        let id = self.dwa.add_state();

        // Compute final weight for this new DWA state
        let mut finalw = Weight::zeros();
        for (sid, cw) in &closure {
            if let Some(fw) = &self.nwa.states[*sid].final_weight {
                let cand = cw & fw;
                if !cand.is_empty() {
                    finalw |= &cand;
                }
            }
        }
        if !finalw.is_empty() {
            let _ = self.dwa.set_final_weight(id, finalw);
        }

        self.seen.insert(closure.clone(), id);
        self.closures.push(closure);
        self.queue.push_back(id);
        id
    }

    fn expand_state(&mut self, sid: usize) {
        let closure = self.closures[sid].clone();
        if closure.is_empty() {
            return;
        }

        // Transitions accumulation: Label -> TargetNWA -> Weight
        // Use BTreeMap for labels to keep them sorted (cleaner DWA), HashMap for targets for speed.
        let mut transitions: BTreeMap<Label, HashMap<NWAStateID, Weight>> = BTreeMap::new();
        let mut edge_weights: HashMap<Label, Weight> = HashMap::new();

        // 1. Collect outgoing labeled transitions from the subset.
        for (u, w_u) in &closure {
            let st = &self.nwa.states[*u];
            for (lbl, targets) in &st.transitions {
                if targets.is_empty() { continue; }

                let target_map = transitions.entry(*lbl).or_default();
                let edge_acc = edge_weights.entry(*lbl).or_insert_with(Weight::zeros);

                for (v, w_trans) in targets {
                    let w_out = w_u & w_trans;
                    if !w_out.is_empty() {
                        *edge_acc |= &w_out;
                        
                        target_map.entry(*v)
                            .and_modify(|w| *w |= &w_out)
                            .or_insert(w_out);
                    }
                }
            }
        }

        // 2. For each label, compute the epsilon-closed destination subset.
        //    We use the precomputed `eps_reach` here.
        for (lbl, raw_targets) in transitions {
            let w_edge = edge_weights.remove(&lbl).unwrap();

            let mut dest_map: HashMap<NWAStateID, Weight> = HashMap::new();

            // Destination = Union_{ t in raw_targets } ( eps_reach[t] intersected with weight(t) )
            for (t, w_t) in raw_targets {
                if t < self.eps_reach.len() {
                    for (v_reach, w_reach) in &self.eps_reach[t] {
                        let combined = &w_t & w_reach;
                        if !combined.is_empty() {
                            dest_map.entry(*v_reach)
                                .and_modify(|w| *w |= &combined)
                                .or_insert(combined);
                        }
                    }
                }
            }

            // Normalize weights in the subset by dividing by w_edge.
            // Division in Boolean semiring (loosening): w / v = w | !v.
            let w_edge_inv = !&w_edge;
            let mut dest_subset: WeightedSubset = dest_map
                .into_iter()
                .map(|(sid, w)| (sid, w | &w_edge_inv))
                .collect();
            dest_subset.sort_unstable_by(|a, b| a.0.cmp(&b.0));

            let dest_dwa_id = self.register_closure(dest_subset);
            let _ = self.dwa.add_transition(sid, lbl, dest_dwa_id, w_edge);
        }
    }
}

/// Precomputes the epsilon closure for every state in the NWA.
fn precompute_all_epsilon_closures(states: &NWAStates) -> Vec<WeightedSubset> {
    let n = states.len();
    let mut reachability = Vec::with_capacity(n);

    for start_node in 0..n {
        let mut dists: HashMap<NWAStateID, Weight> = HashMap::new();
        let mut queue: VecDeque<NWAStateID> = VecDeque::new();

        // Self-reachability is identity
        dists.insert(start_node, Weight::all());
        queue.push_back(start_node);

        while let Some(u) = queue.pop_front() {
            let w_u = dists.get(&u).unwrap().clone();
            
            if u < n {
                for (v, w_eps) in &states[u].epsilons {
                    let new_w = &w_u & w_eps;
                    if new_w.is_empty() { continue; }

                    let entry = dists.entry(*v).or_insert_with(Weight::zeros);
                    if !new_w.is_subset_of(entry) {
                        *entry |= &new_w;
                        queue.push_back(*v);
                    }
                }
            }
        }

        let mut sub: WeightedSubset = dists.into_iter().collect();
        sub.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        reachability.push(sub);
    }

    reachability
}

/// Computes epsilon closure for a specific subset on the fly.
/// Used by the heuristic singleton check.
fn epsilon_closure_optimized(nwa_states: &NWAStates, seed: &WeightedSubset) -> WeightedSubset {
    let mut closure_map: HashMap<NWAStateID, Weight> = HashMap::with_capacity(seed.len() * 2);
    let mut queue: VecDeque<NWAStateID> = VecDeque::with_capacity(seed.len());

    for (sid, w) in seed {
        if !is_zero(w) {
            closure_map.insert(*sid, w.clone());
            queue.push_back(*sid);
        }
    }

    while let Some(u) = queue.pop_front() {
        let uw = if let Some(w) = closure_map.get(&u) {
            w.clone()
        } else {
            continue;
        };

        if u >= nwa_states.len() { continue; }
        
        for (v, w_eps) in &nwa_states[u].epsilons {
            let cand = &uw & w_eps;
            if cand.is_empty() { continue; }

            let entry = closure_map.entry(*v).or_insert_with(Weight::zeros);
            if !cand.is_subset_of(entry) {
                *entry |= &cand;
                queue.push_back(*v);
            }
        }
    }

    let mut result: Vec<(NWAStateID, Weight)> = closure_map.into_iter().collect();
    result.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    result
}

/// Heuristic optimization for single-state loop unions.
fn try_build_singleton_loop_union(nwa: &NWA) -> Option<DWA> {
    if nwa.states.0.is_empty() || nwa.body.start_states.len() != 1 {
        return None;
    }

    let start = nwa.body.start_states[0];
    if start >= nwa.states.len() { return None; }

    if !nwa.states[start].transitions.is_empty() {
        return None;
    }

    let mut seed: WeightedSubset = Vec::new();
    seed.push((start, Weight::all()));
    // Use the local helper here to avoid precomputing everything for this fast path
    let start_closure = epsilon_closure_optimized(&nwa.states, &seed);

    let mut comps: Vec<(NWAStateID, Weight)> = Vec::new();
    for (sid, cw) in start_closure.iter() {
        if *sid == start || is_zero(cw) {
            continue;
        }
        let st = &nwa.states[*sid];

        if !st.epsilons.is_empty() {
            return None;
        }
        for (_lbl, vec_targets) in st.transitions.iter() {
            for (to, _) in vec_targets {
                if *to != *sid {
                    return None;
                }
            }
        }

        if let Some(fw) = &st.final_weight {
            let base = cw & fw;
            if !base.is_empty() {
                comps.push((*sid, base));
            }
        }
    }

    if comps.is_empty() {
        return None;
    }

    for i in 0..comps.len() {
        for j in (i + 1)..comps.len() {
            if !(comps[i].1.clone() & comps[j].1.clone()).is_empty() {
                return None;
            }
        }
    }

    let mut label_to_weight: BTreeMap<Label, Weight> = BTreeMap::new();
    for (sid, base) in &comps {
        let st = &nwa.states[*sid];
        for (lbl, vec_targets) in st.transitions.iter() {
            let mut w_union = Weight::zeros();
            for (_to, w) in vec_targets {
                w_union = w_union | w.clone();
            }
            if !w_union.is_empty() {
                let contrib = base.clone() & w_union;
                if !contrib.is_empty() {
                    let prev = label_to_weight.get(lbl).cloned().unwrap_or_else(Weight::zeros);
                    label_to_weight.insert(*lbl, prev | contrib);
                }
            }
        }
    }

    let mut final_union = Weight::zeros();
    for (_sid, base) in &comps {
        final_union = final_union | base.clone();
    }

    let mut dwa = DWA::new();
    let s0 = dwa.body.start_state;
    if !final_union.is_empty() {
        let _ = dwa.set_final_weight(s0, final_union);
    }
    for (lbl, w) in label_to_weight {
        if !w.is_empty() {
            let _ = dwa.add_transition(s0, lbl, s0, w);
        }
    }

    Some(dwa)
}