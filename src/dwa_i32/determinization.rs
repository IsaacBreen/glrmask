#![allow(dead_code)]
#![allow(clippy::needless_borrow)]

use rustc_hash::FxHashMap;
use std::cell::Cell;
use std::cmp::Ordering;
use std::collections::{BTreeMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::ops::BitOrAssign;
use std::sync::Arc;
use std::time::{Duration, Instant};
use profiler_macro::timeit;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use crate::dwa_i32::test_weighted_automata;
use super::common::{DETERMINIZE_DEBUG, Label, NWAStateID, Weight};
use super::determinization_acyclic::topo_order_if_acyclic;
use super::determinization_cyclic::precompute_all_epsilon_closures;
use super::dwa::DWA;
use super::nwa::{NWA, NWAStates};

// Global counter for determinizations
static DETERMINIZE_COUNT: AtomicUsize = AtomicUsize::new(0);
static TOTAL_DETERMINIZE_TIME_MS: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static DETERMINIZE_PROGRESS_ENABLED: Cell<bool> = Cell::new(false);
}

pub(crate) fn with_determinize_progress_enabled<R>(enabled: bool, f: impl FnOnce() -> R) -> R {
    DETERMINIZE_PROGRESS_ENABLED.with(|flag| {
        let prev = flag.get();
        flag.set(enabled);
        let result = f();
        flag.set(prev);
        result
    })
}

fn determinize_progress_enabled() -> bool {
    DETERMINIZE_PROGRESS_ENABLED.with(|flag| flag.get())
}

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
pub(crate) type WeightedSubset = Vec<(NWAStateID, Weight)>;

fn is_zero(w: &Weight) -> bool { w.is_empty() }

/// A pre-hashed wrapper for a weighted subset using sorted Vec for fast iteration.
#[derive(Clone)]
pub(crate) struct HashedSubset {
    inner: Vec<(NWAStateID, Weight)>,  // Sorted by NWAStateID
    hash: u64,
}

impl HashedSubset {
    pub(crate) fn from_btreemap(map: BTreeMap<NWAStateID, Weight>) -> Self {
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
    
    pub(crate) fn from_fxhashmap(map: FxHashMap<NWAStateID, Weight>) -> Self {
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

    pub(crate) fn from_sorted_vec(inner: WeightedSubset) -> Self {
        use rustc_hash::FxHasher;
        let mut hasher = FxHasher::default();
        for (k, v) in &inner {
            k.hash(&mut hasher);
            v.hash(&mut hasher);
        }
        let hash = hasher.finish();
        Self { inner, hash }
    }

    pub(crate) fn new(inner: BTreeMap<NWAStateID, Weight>) -> Self {
        Self::from_btreemap(inner)
    }
    
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    pub(crate) fn iter(&self) -> std::slice::Iter<'_, (NWAStateID, Weight)> {
        self.inner.iter()
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
    /// - Displays a progress bar for large automata.
    /// - **Formerly:** `determinize_to_dwa2`
    pub fn determinize_robust(&self) -> DWA {
        // Handle empty NWA early
        if self.states.0.is_empty() {
            return DWA::new();
        }
        
        // 0. Optionally remove epsilons first via rustfst (produces more compact DWA)
        // This matches what determinize_to_dwa_with_rustfst does.
        let use_rm_epsilon = std::env::var("DWA_USE_RM_EPSILON").map_or(false, |v| v == "1");
        if use_rm_epsilon {
            crate::debug!(5, "Determinization: Removing epsilons via rustfst first...");
            let epsilon_free = self.remove_epsilons();
            return epsilon_free.determinize_robust_internal();
        }
        
        self.determinize_robust_internal()
    }
    
    /// Internal implementation of determinize_robust (called after optional rm_epsilon).
    fn determinize_robust_internal(&self) -> DWA {
        // 1. Try Heuristic Optimization
        if let Some(dwa) = try_build_singleton_loop_union(self) {
            return dwa;
        }

        // 2. Setup
        if self.states.0.is_empty() {
            return DWA::new();
        }

        let macro_level = std::env::var("MACRO_DEBUG_LEVEL")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(0);
        let profile_enabled = std::env::var("PROFILE_DETERMINIZATION_BREAKDOWN")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
            || macro_level >= 5;
        let progress_enabled = determinize_progress_enabled() && macro_level >= 4;

        if let Some(topo_order) = topo_order_if_acyclic(self) {
            crate::debug!(6, "Determinization: Using acyclic fast path...");
            let dwa = super::determinization_acyclic::determinize_acyclic_with_progress(self, &topo_order, progress_enabled);
            if DETERMINIZE_DEBUG {
                let rustfst_dwa = self.determinize_to_dwa_with_rustfst();
                crate::debug!(5, "[DETERMINIZE_DEBUG] Comparing custom determinization with rustfst...");
                test_weighted_automata::stochastic_equivalence_test(dwa.clone(), rustfst_dwa);
            }
            return dwa;
        }

        crate::debug!(6, "Determinization: Precomputing epsilon closures...");

        // 3. Precompute Reachability (cyclic)
        let eps_start = if profile_enabled { Some(Instant::now()) } else { None };
        let eps_reach = precompute_all_epsilon_closures(&self.states);

        // 4. Initialize Determinizer
        let mut det = Determinizer::new(self, &eps_reach, profile_enabled, progress_enabled);
        if let Some(start) = eps_start {
            det.profile.precompute_eps = start.elapsed();
        }

        // 5. Initial State Construction
        let start_subset_start = if profile_enabled { Some(Instant::now()) } else { None };
        let mut start_map: FxHashMap<NWAStateID, Weight> = FxHashMap::default();
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
        if let Some(start) = start_subset_start {
            det.profile.start_subset = start.elapsed();
        }

        // 6. Main Expansion Loop
        let mut processed_subsets = 0usize;
        let progress_start = Instant::now();
        let mut progress_last_log = Instant::now();
        let progress_log_every = 5_000usize;
        let progress_log_interval = Duration::from_secs(2);
        while let Some(sid) = det.queue.pop_front() {
            let expand_start = if profile_enabled { Some(Instant::now()) } else { None };
            timeit!("determinize::expand_state", {
                det.expand_state(sid);
            });
            processed_subsets += 1;
            if let Some(start) = expand_start {
                det.profile.expand_total += start.elapsed();
            }
            if progress_enabled
                && (processed_subsets % progress_log_every == 0
                    || progress_last_log.elapsed() >= progress_log_interval)
            {
                crate::debug!(
                    4,
                    "Determinize progress: subsets_processed={}, dwa_states={}, queue={}, transitions_added={}, elapsed={:?}",
                    processed_subsets,
                    det.dwa.states.len(),
                    det.queue.len(),
                    det.progress_transitions,
                    progress_start.elapsed(),
                );
                progress_last_log = Instant::now();
            }
        }

        if progress_enabled {
            crate::debug!(
                4,
                "Determinize complete: subsets_processed={}, dwa_states={}, transitions_added={}, elapsed={:?}",
                processed_subsets,
                det.dwa.states.len(),
                det.progress_transitions,
                progress_start.elapsed(),
            );
        }

        if profile_enabled {
            det.profile.log();
        }

        // 7. Debug Verification
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
    seen: FxHashMap<Arc<HashedSubset>, NWAStateID>,
    queue: VecDeque<usize>,
    // Store the closure for each DWA state
    closures: Vec<Arc<HashedSubset>>,
    
    dwa: DWA,

    profile_enabled: bool,
    profile: DeterminizeProfile,
    progress_enabled: bool,
    progress_transitions: usize,
}

#[derive(Default)]
struct DeterminizeProfile {
    precompute_eps: Duration,
    start_subset: Duration,
    expand_total: Duration,
    collect_transitions: Duration,
    collect_weight_ops: Duration,
    collect_map_ops: Duration,
    collect_and: Duration,
    collect_or: Duration,
    collect_clone: Duration,
    collect_state_iters: u64,
    collect_label_iters: u64,
    collect_target_iters: u64,
    collect_btree_entry: Duration,
    collect_btree_hits: u64,
    collect_btree_misses: u64,
    collect_edge_entry: Duration,
    collect_edge_hits: u64,
    collect_edge_misses: u64,
    collect_target_lookup: Duration,
    collect_target_hits: u64,
    collect_target_misses: u64,
    collect_target_insert: Duration,
    collect_target_insert_count: u64,
    collect_and_count: u64,
    collect_or_count: u64,
    build_dest_map: Duration,
    build_dest_map_weight_ops: Duration,
    build_dest_map_map_ops: Duration,
    build_dest_map_and: Duration,
    build_dest_map_or: Duration,
    build_dest_map_targets: u64,
    build_dest_map_pairs: u64,
    build_dest_map_lookup: Duration,
    build_dest_map_hits: u64,
    build_dest_map_misses: u64,
    build_dest_map_insert: Duration,
    build_dest_map_insert_count: u64,
    build_dest_map_and_count: u64,
    build_dest_map_or_count: u64,
    normalize_subset: Duration,
    normalize_invert: Duration,
    normalize_build: Duration,
    normalize_sort: Duration,
    normalize_not: Duration,
    normalize_or: Duration,
    normalize_vec_alloc: Duration,
    normalize_push: Duration,
    normalize_push_count: u64,
    normalize_items: u64,
    normalize_not_count: u64,
    normalize_or_count: u64,
    normalize_sort_len: u64,
    register_closure: Duration,
    register_lookup: Duration,
    register_add_state: Duration,
    register_clone: Duration,
    register_insert: Duration,
    register_push_closure: Duration,
    register_push_queue: Duration,
    final_weight: Duration,
    register_and: Duration,
    register_or: Duration,
    add_transition: Duration,
    expand_calls: u64,
    labels: u64,
    raw_targets: u64,
    eps_pairs: u64,
    closure_size_total: u64,
}

impl DeterminizeProfile {
    fn log(&self) {
        if self.expand_calls == 0 {
            return;
        }
        let avg_closure_size = self.closure_size_total as f64 / self.expand_calls as f64;
        eprintln!("TIMING: determinize::precompute_eps {:?}", self.precompute_eps);
        eprintln!("TIMING: determinize::start_subset {:?}", self.start_subset);
        eprintln!(
            "TIMING: determinize::expand_total {:?} over {} expansions (avg_closure_size={:.2})",
            self.expand_total,
            self.expand_calls,
            avg_closure_size,
        );
        eprintln!("TIMING: determinize::collect_transitions {:?}", self.collect_transitions);
        eprintln!("TIMING: determinize::build_dest_map {:?}", self.build_dest_map);
        eprintln!("TIMING: determinize::normalize_subset {:?}", self.normalize_subset);
        eprintln!("TIMING: determinize::register_closure {:?}", self.register_closure);
        eprintln!("TIMING: determinize::add_transition {:?}", self.add_transition);
        eprintln!(
            "TIMING: determinize::counters labels={} raw_targets={} eps_pairs={}",
            self.labels,
            self.raw_targets,
            self.eps_pairs,
        );
    }
}

impl<'a> Determinizer<'a> {
    fn new(nwa: &'a NWA, eps_reach: &'a [WeightedSubset], profile_enabled: bool, progress_enabled: bool) -> Self {
        let mut dwa = DWA::new();
        dwa.states.0.clear();
        dwa.body.start_state = 0;
        Determinizer {
            nwa,
            eps_reach,
            seen: FxHashMap::default(),
            queue: VecDeque::new(),
            closures: Vec::new(),
            dwa,
            profile_enabled,
            profile: DeterminizeProfile::default(),
            progress_enabled,
            progress_transitions: 0,
        }
    }

    fn register_closure(&mut self, closure: WeightedSubset) -> usize {
        let register_start = if self.profile_enabled { Some(Instant::now()) } else { None };
        let hashed = HashedSubset::from_sorted_vec(closure);
        let lookup_start = if self.profile_enabled { Some(Instant::now()) } else { None };
        if let Some(&id) = self.seen.get(&hashed) {
            if let Some(start) = lookup_start {
                self.profile.register_lookup += start.elapsed();
            }
            if let Some(start) = register_start {
                self.profile.register_closure += start.elapsed();
            }
            return id;
        }
        if let Some(start) = lookup_start {
            self.profile.register_lookup += start.elapsed();
        }

        let add_state_start = if self.profile_enabled { Some(Instant::now()) } else { None };
        let id = self.dwa.add_state();
        if let Some(start) = add_state_start {
            self.profile.register_add_state += start.elapsed();
        }

        // Compute final weight for this new DWA state
        let final_weight_start = if self.profile_enabled { Some(Instant::now()) } else { None };
        let mut finalw = Weight::zeros();
        for (sid, cw) in &hashed.inner {
            if let Some(fw) = &self.nwa.states[*sid].final_weight {
                let and_start = if self.profile_enabled { Some(Instant::now()) } else { None };
                let cand = cw & fw;
                if let Some(start) = and_start {
                    let elapsed = start.elapsed();
                    self.profile.register_and += elapsed;
                }
                if !cand.is_empty() {
                    let or_start = if self.profile_enabled { Some(Instant::now()) } else { None };
                    finalw |= &cand;
                    if let Some(start) = or_start {
                        let elapsed = start.elapsed();
                        self.profile.register_or += elapsed;
                    }
                }
            }
        }
        if !finalw.is_empty() {
            let _ = self.dwa.set_final_weight(id, finalw);
        }
        if let Some(start) = final_weight_start {
            self.profile.final_weight += start.elapsed();
        }

        let hashed = Arc::new(hashed);
        let clone_start = if self.profile_enabled { Some(Instant::now()) } else { None };
        let closure_clone = Arc::clone(&hashed);
        if let Some(start) = clone_start {
            self.profile.register_clone += start.elapsed();
        }

        let insert_start = if self.profile_enabled { Some(Instant::now()) } else { None };
        self.seen.insert(closure_clone, id);
        if let Some(start) = insert_start {
            self.profile.register_insert += start.elapsed();
        }

        let push_closure_start = if self.profile_enabled { Some(Instant::now()) } else { None };
        self.closures.push(hashed);
        if let Some(start) = push_closure_start {
            self.profile.register_push_closure += start.elapsed();
        }

        let push_queue_start = if self.profile_enabled { Some(Instant::now()) } else { None };
        self.queue.push_back(id);
        if let Some(start) = push_queue_start {
            self.profile.register_push_queue += start.elapsed();
        }

        if let Some(start) = register_start {
            self.profile.register_closure += start.elapsed();
        }
        id
    }

    fn expand_state(&mut self, sid: usize) {
        if self.profile_enabled {
            self.profile.expand_calls += 1;
        }
        let closure_clone_start = if self.profile_enabled { Some(Instant::now()) } else { None };
        let closure = self.closures[sid].clone();
        if let Some(start) = closure_clone_start {
            self.profile.collect_clone += start.elapsed();
        }
        if closure.is_empty() {
            return;
        }
        if self.profile_enabled {
            self.profile.closure_size_total += closure.len() as u64;
        }

        // Transitions accumulation: Label -> TargetNWA -> Weight
        // Use BTreeMap for labels to keep them sorted (cleaner DWA), HashMap for targets for speed.
        let mut transitions: BTreeMap<Label, FxHashMap<NWAStateID, Weight>> = BTreeMap::new();
        let mut edge_weights: FxHashMap<Label, Weight> = FxHashMap::default();

        // 1. Collect outgoing labeled transitions from the subset.
        let collect_start = if self.profile_enabled { Some(Instant::now()) } else { None };
        timeit!("determinize::collect_transitions", {
            for (u, w_u) in &closure.inner {
                if self.profile_enabled {
                    self.profile.collect_state_iters += 1;
                }
                let st = &self.nwa.states[*u];
                for (lbl, targets) in &st.transitions {
                    if targets.is_empty() { continue; }

                    if self.profile_enabled {
                        self.profile.collect_label_iters += 1;
                    }

                    let target_map = timeit!("determinize::collect_transitions::label_grouping", {
                        if self.profile_enabled {
                            let entry_start = Instant::now();
                            let entry = transitions.entry(*lbl);
                            let (map_ref, is_new) = match entry {
                                std::collections::btree_map::Entry::Occupied(o) => (o.into_mut(), false),
                                std::collections::btree_map::Entry::Vacant(v) => (v.insert(FxHashMap::default()), true),
                            };
                            let elapsed = entry_start.elapsed();
                            self.profile.collect_btree_entry += elapsed;
                            self.profile.collect_map_ops += elapsed;
                            if is_new {
                                self.profile.collect_btree_misses += 1;
                            } else {
                                self.profile.collect_btree_hits += 1;
                            }
                            map_ref
                        } else {
                            transitions.entry(*lbl).or_default()
                        }
                    });

                    let edge_acc = timeit!("determinize::collect_transitions::edge_acc", {
                        if self.profile_enabled {
                            let entry_start = Instant::now();
                            let entry = edge_weights.entry(*lbl);
                            let (acc_ref, is_new) = match entry {
                                std::collections::hash_map::Entry::Occupied(o) => (o.into_mut(), false),
                                std::collections::hash_map::Entry::Vacant(v) => (v.insert(Weight::zeros()), true),
                            };
                            let elapsed = entry_start.elapsed();
                            self.profile.collect_edge_entry += elapsed;
                            self.profile.collect_map_ops += elapsed;
                            if is_new {
                                self.profile.collect_edge_misses += 1;
                            } else {
                                self.profile.collect_edge_hits += 1;
                            }
                            acc_ref
                        } else {
                            edge_weights.entry(*lbl).or_insert_with(Weight::zeros)
                        }
                    });

                    timeit!("determinize::collect_transitions::target_merge", {
                        for (v, w_trans) in targets {
                            if self.profile_enabled {
                                self.profile.collect_target_iters += 1;
                            }
                            let and_start = if self.profile_enabled { Some(Instant::now()) } else { None };
                            let w_out = timeit!("determinize::collect_transitions::weight_and", {
                                w_u & w_trans
                            });
                            if let Some(start) = and_start {
                                let elapsed = start.elapsed();
                                self.profile.collect_weight_ops += elapsed;
                                self.profile.collect_and += elapsed;
                                self.profile.collect_and_count += 1;
                            }
                            if !w_out.is_empty() {
                                let or_start = if self.profile_enabled { Some(Instant::now()) } else { None };
                                timeit!("determinize::collect_transitions::weight_or", {
                                    *edge_acc |= &w_out;
                                });
                                if let Some(start) = or_start {
                                    let elapsed = start.elapsed();
                                    self.profile.collect_weight_ops += elapsed;
                                    self.profile.collect_or += elapsed;
                                    self.profile.collect_or_count += 1;
                                }

                                if self.profile_enabled {
                                    let lookup_start = Instant::now();
                                    if let Some(existing) = target_map.get_mut(v) {
                                        let lookup_elapsed = lookup_start.elapsed();
                                        self.profile.collect_target_lookup += lookup_elapsed;
                                        self.profile.collect_map_ops += lookup_elapsed;
                                        self.profile.collect_target_hits += 1;

                                        let inner_or_start = Instant::now();
                                        timeit!("determinize::collect_transitions::weight_or", {
                                            *existing |= &w_out;
                                        });
                                        let inner_or_elapsed = inner_or_start.elapsed();
                                        self.profile.collect_weight_ops += inner_or_elapsed;
                                        self.profile.collect_or += inner_or_elapsed;
                                        self.profile.collect_or_count += 1;
                                    } else {
                                        let lookup_elapsed = lookup_start.elapsed();
                                        self.profile.collect_target_lookup += lookup_elapsed;
                                        self.profile.collect_map_ops += lookup_elapsed;
                                        self.profile.collect_target_misses += 1;

                                        let insert_start = Instant::now();
                                        target_map.insert(*v, w_out);
                                        let insert_elapsed = insert_start.elapsed();
                                        self.profile.collect_target_insert += insert_elapsed;
                                        self.profile.collect_map_ops += insert_elapsed;
                                        self.profile.collect_target_insert_count += 1;
                                    }
                                } else if let Some(existing) = target_map.get_mut(v) {
                                    *existing |= &w_out;
                                } else {
                                    target_map.insert(*v, w_out);
                                }
                            }
                        }
                    });
                }
            }
        });
        if let Some(start) = collect_start {
            self.profile.collect_transitions += start.elapsed();
        }

        // 2. For each label, compute the epsilon-closed destination subset.
        //    We use the precomputed `eps_reach` here.
        timeit!("determinize::build_destinations", {
            for (lbl, raw_targets) in transitions {
            let w_edge = edge_weights.remove(&lbl).unwrap();

            if self.profile_enabled {
                self.profile.labels += 1;
                self.profile.raw_targets += raw_targets.len() as u64;
            }

            let mut dest_map: FxHashMap<NWAStateID, Weight> = FxHashMap::default();

            // Destination = Union_{ t in raw_targets } ( eps_reach[t] intersected with weight(t) )
            let dest_map_start = if self.profile_enabled { Some(Instant::now()) } else { None };
            timeit!("determinize::build_destinations::build_dest_map", {
                for (t, w_t) in raw_targets {
                    if self.profile_enabled {
                        self.profile.build_dest_map_targets += 1;
                    }
                    if t < self.eps_reach.len() {
                        if self.profile_enabled {
                            let reach_len = self.eps_reach[t].len() as u64;
                            self.profile.eps_pairs += reach_len;
                            self.profile.build_dest_map_pairs += reach_len;
                        }
                        for (v_reach, w_reach) in &self.eps_reach[t] {
                            let and_start = if self.profile_enabled { Some(Instant::now()) } else { None };
                            let combined = timeit!("determinize::build_destinations::weight_and", {
                                &w_t & w_reach
                            });
                            if let Some(start) = and_start {
                                let elapsed = start.elapsed();
                                self.profile.build_dest_map_weight_ops += elapsed;
                                self.profile.build_dest_map_and += elapsed;
                                self.profile.build_dest_map_and_count += 1;
                            }
                            if !combined.is_empty() {
                                if self.profile_enabled {
                                    let lookup_start = Instant::now();
                                    if let Some(existing) = dest_map.get_mut(v_reach) {
                                        let lookup_elapsed = lookup_start.elapsed();
                                        self.profile.build_dest_map_lookup += lookup_elapsed;
                                        self.profile.build_dest_map_map_ops += lookup_elapsed;
                                        self.profile.build_dest_map_hits += 1;

                                        let or_start = Instant::now();
                                        timeit!("determinize::build_destinations::weight_or", {
                                            *existing |= &combined;
                                        });
                                        let or_elapsed = or_start.elapsed();
                                        self.profile.build_dest_map_weight_ops += or_elapsed;
                                        self.profile.build_dest_map_or += or_elapsed;
                                        self.profile.build_dest_map_or_count += 1;
                                    } else {
                                        let lookup_elapsed = lookup_start.elapsed();
                                        self.profile.build_dest_map_lookup += lookup_elapsed;
                                        self.profile.build_dest_map_map_ops += lookup_elapsed;
                                        self.profile.build_dest_map_misses += 1;

                                        let insert_start = Instant::now();
                                        dest_map.insert(*v_reach, combined);
                                        let insert_elapsed = insert_start.elapsed();
                                        self.profile.build_dest_map_insert += insert_elapsed;
                                        self.profile.build_dest_map_map_ops += insert_elapsed;
                                        self.profile.build_dest_map_insert_count += 1;
                                    }
                                } else if let Some(existing) = dest_map.get_mut(v_reach) {
                                    *existing |= &combined;
                                } else {
                                    dest_map.insert(*v_reach, combined);
                                }
                            }
                        }
                    }
                }
            });
            if let Some(start) = dest_map_start {
                self.profile.build_dest_map += start.elapsed();
            }

            // Normalize weights in the subset by dividing by w_edge.
            // Division in Boolean semiring (loosening): w / v = w | !v.
            let dest_subset = timeit!("determinize::build_destinations::normalize_subset", {
                let normalize_start = if self.profile_enabled { Some(Instant::now()) } else { None };
                let invert_start = if self.profile_enabled { Some(Instant::now()) } else { None };
                let w_edge_inv = !&w_edge;
                if let Some(start) = invert_start {
                    let elapsed = start.elapsed();
                    self.profile.normalize_invert += elapsed;
                    self.profile.normalize_not += elapsed;
                    self.profile.normalize_not_count += 1;
                }
                let build_start = if self.profile_enabled { Some(Instant::now()) } else { None };
                if self.profile_enabled {
                    self.profile.normalize_items += dest_map.len() as u64;
                }
                let vec_alloc_start = if self.profile_enabled { Some(Instant::now()) } else { None };
                let mut dest_subset: WeightedSubset = Vec::with_capacity(dest_map.len());
                if let Some(start) = vec_alloc_start {
                    self.profile.normalize_vec_alloc += start.elapsed();
                }
                if self.profile_enabled {
                    for (sid, w) in dest_map {
                        let or_start = Instant::now();
                        let combined = w | &w_edge_inv;
                        let elapsed = or_start.elapsed();
                        self.profile.normalize_or += elapsed;
                        self.profile.normalize_or_count += 1;
                        let push_start = Instant::now();
                        dest_subset.push((sid, combined));
                        let push_elapsed = push_start.elapsed();
                        self.profile.normalize_push += push_elapsed;
                        self.profile.normalize_push_count += 1;
                    }
                } else {
                    for (sid, w) in dest_map {
                        dest_subset.push((sid, w | &w_edge_inv));
                    }
                }
                if let Some(start) = build_start {
                    self.profile.normalize_build += start.elapsed();
                }
                let sort_start = if self.profile_enabled { Some(Instant::now()) } else { None };
                if self.profile_enabled {
                    self.profile.normalize_sort_len += dest_subset.len() as u64;
                }
                dest_subset.sort_unstable_by(|a, b| a.0.cmp(&b.0));
                if let Some(start) = sort_start {
                    self.profile.normalize_sort += start.elapsed();
                }
                if let Some(start) = normalize_start {
                    self.profile.normalize_subset += start.elapsed();
                }
                dest_subset
            });

            let dest_dwa_id = timeit!("determinize::build_destinations::register_closure", {
                self.register_closure(dest_subset)
            });
            let add_start = if self.profile_enabled { Some(Instant::now()) } else { None };
            let _ = self.dwa.add_transition(sid, lbl, dest_dwa_id, w_edge);
            if self.progress_enabled {
                self.progress_transitions += 1;
            }
            if let Some(start) = add_start {
                self.profile.add_transition += start.elapsed();
            }
            }
        });
    }
}

/// Computes epsilon closure for a specific subset on the fly.
/// Used by the heuristic singleton check.
fn epsilon_closure_optimized(nwa_states: &NWAStates, seed: &WeightedSubset) -> WeightedSubset {
    let mut closure_map: FxHashMap<NWAStateID, Weight> = FxHashMap::default();
    closure_map.reserve(seed.len() * 2);
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