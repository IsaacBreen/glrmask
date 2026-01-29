use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::env;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering as AtomicOrdering};
use std::time::{Duration, Instant};

use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use profiler_macro::{time_it, timeit};

use super::common::{Label, NWAStateID, Weight};
use super::determinization::{HashedSubset, WeightedSubset};
use super::dwa::{DWA, DWABody, DWAState, DWAStates};
use super::nwa::{NWA, NWAStates};

fn percentile_index(len: usize, percentile: f64) -> usize {
    if len == 0 {
        0
    } else {
        ((len as f64 - 1.0) * percentile).round() as usize
    }
}

pub(crate) fn topo_order_if_acyclic(nwa: &NWA) -> Option<Vec<usize>> {
    let n = nwa.states.len();
    if n == 0 {
        return Some(Vec::new());
    }

    let mut indegree = vec![0usize; n];
    for st in &nwa.states.0 {
        for (v, _w) in &st.epsilons {
            if *v < n {
                indegree[*v] += 1;
            }
        }
        for targets in st.transitions.values() {
            for (v, _w) in targets {
                if *v < n {
                    indegree[*v] += 1;
                }
            }
        }
    }

    let mut queue: VecDeque<usize> = indegree
        .iter()
        .enumerate()
        .filter_map(|(i, &deg)| if deg == 0 { Some(i) } else { None })
        .collect();

    let mut order = Vec::with_capacity(n);
    while let Some(u) = queue.pop_front() {
        order.push(u);
        let st = &nwa.states[u];
        for (v, _w) in &st.epsilons {
            if *v >= n {
                continue;
            }
            indegree[*v] = indegree[*v].saturating_sub(1);
            if indegree[*v] == 0 {
                queue.push_back(*v);
            }
        }
        for targets in st.transitions.values() {
            for (v, _w) in targets {
                if *v >= n {
                    continue;
                }
                indegree[*v] = indegree[*v].saturating_sub(1);
                if indegree[*v] == 0 {
                    queue.push_back(*v);
                }
            }
        }
    }

    if order.len() == n {
        Some(order)
    } else {
        None
    }
}

pub(crate) fn precompute_all_epsilon_closures_acyclic(states: &NWAStates, topo: &[usize]) -> Vec<WeightedSubset> {
    let n = states.len();
    let mut closure_maps: Vec<FxHashMap<NWAStateID, Weight>> = (0..n)
        .map(|_| FxHashMap::default())
        .collect();

    for &u in topo.iter().rev() {
        let mut closure: FxHashMap<NWAStateID, Weight> = FxHashMap::default();
        closure.insert(u, Weight::all());

        for (v, w_uv) in &states[u].epsilons {
            if *v >= n {
                continue;
            }
            for (t, w_vt) in &closure_maps[*v] {
                let combined = w_uv & w_vt;
                if combined.is_empty() {
                    continue;
                }
                let entry = closure.entry(*t).or_insert_with(Weight::zeros);
                if !combined.is_subset_of(entry) {
                    *entry |= &combined;
                }
            }
        }

        closure_maps[u] = closure;
    }

    closure_maps
        .into_iter()
        .map(|map| {
            let mut vec: WeightedSubset = map.into_iter().collect();
            vec.sort_unstable_by(|a, b| a.0.cmp(&b.0));
            vec
        })
        .collect()
}

pub(crate) fn precompute_all_epsilon_closures_acyclic_unweighted(
    states: &NWAStates,
    topo: &[usize],
) -> Vec<BTreeSet<NWAStateID>> {
    let n = states.len();
    let mut closure_sets: Vec<BTreeSet<NWAStateID>> = (0..n)
        .map(|_| BTreeSet::new())
        .collect();

    for &u in topo.iter().rev() {
        let mut closure: BTreeSet<NWAStateID> = BTreeSet::new();
        closure.insert(u);

        for (v, _w_uv) in &states[u].epsilons {
            if *v >= n {
                continue;
            }
            for &t in &closure_sets[*v] {
                closure.insert(t);
            }
        }

        closure_sets[u] = closure;
    }

    closure_sets
}

fn build_destinations_unweighted(
    closure: &BTreeSet<NWAStateID>,
    nwa: &NWA,
    eps_reach: &[BTreeSet<NWAStateID>],
) -> Vec<(Label, BTreeSet<NWAStateID>)> {
    let mut transitions: BTreeMap<Label, BTreeSet<NWAStateID>> = BTreeMap::new();

    for u in closure {
        let st = &nwa.states[*u];
        for (lbl, targets) in &st.transitions {
            let entry = transitions.entry(*lbl).or_default();
            for (v, _w_uv) in targets {
                entry.insert(*v);
            }
        }
    }

    let mut results = Vec::new();
    for (lbl, dest_set) in transitions {
        let mut expanded: BTreeSet<NWAStateID> = BTreeSet::new();
        for v in dest_set {
            if v >= eps_reach.len() {
                continue;
            }
            expanded.extend(eps_reach[v].iter().copied());
        }
        if expanded.is_empty() {
            continue;
        }
        results.push((lbl, expanded));
    }

    results
}

fn compute_unweighted_dwa_topo_order(
    transition_cache: &[Vec<(Label, usize)>],
) -> Vec<usize> {
    let total = transition_cache.len();
    let mut indegree = vec![0usize; total];
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); total];

    for (sid, transitions) in transition_cache.iter().enumerate() {
        let mut targets: FxHashSet<usize> = FxHashSet::default();
        for &(_lbl, dest_id) in transitions {
            targets.insert(dest_id);
        }
        for dest_id in targets {
            adj[sid].push(dest_id);
            indegree[dest_id] = indegree[dest_id].saturating_add(1);
        }
    }

    let mut queue: VecDeque<usize> = indegree
        .iter()
        .enumerate()
        .filter_map(|(i, &deg)| if deg == 0 { Some(i) } else { None })
        .collect();

    let mut order = Vec::with_capacity(total);
    while let Some(u) = queue.pop_front() {
        order.push(u);
        for &v in &adj[u] {
            indegree[v] = indegree[v].saturating_sub(1);
            if indegree[v] == 0 {
                queue.push_back(v);
            }
        }
    }

    if order.len() == total {
        order
    } else {
        (0..total).collect()
    }
}

#[time_it]
fn build_destinations_batched(
    closure: &FxHashMap<NWAStateID, Weight>,
    nwa: &NWA,
    eps_reach: &[WeightedSubset],
    timers: Option<&MaterializeTimers>,
) -> Vec<(Label, WeightedSubset, Weight)> {
    if let Some(timers) = timers {
        timers.calls.fetch_add(1, AtomicOrdering::Relaxed);
        timers
            .closure_size_total
            .fetch_add(closure.len() as u64, AtomicOrdering::Relaxed);
    }

    let mut transitions: FxHashMap<Label, FxHashMap<NWAStateID, Weight>> = FxHashMap::default();
    let mut edge_weights: FxHashMap<Label, Weight> = FxHashMap::default();

    let collect_start = if timers.is_some() { Some(Instant::now()) } else { None };
    for (u, w_u) in closure {
        let st = &nwa.states[*u];
        if let Some(timers) = timers {
            timers
                .transition_total
                .fetch_add(st.transitions.len() as u64, AtomicOrdering::Relaxed);
        }
        for (lbl, targets) in &st.transitions {
            if let Some(timers) = timers {
                timers
                    .target_total
                    .fetch_add(targets.len() as u64, AtomicOrdering::Relaxed);
            }
            for (v, w_uv) in targets {
                let combined = if let Some(timers) = timers {
                    let start = Instant::now();
                    let combined = w_u & w_uv;
                    timers
                        .and_ns
                        .fetch_add(start.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);
                    timers
                        .and_ops
                        .fetch_add(1, AtomicOrdering::Relaxed);
                    combined
                } else {
                    w_u & w_uv
                };
                if combined.is_empty() {
                    continue;
                }
                let edge_entry = edge_weights.entry(*lbl).or_insert_with(Weight::zeros);
                if let Some(timers) = timers {
                    let start = Instant::now();
                    *edge_entry |= &combined;
                    timers
                        .or_ns
                        .fetch_add(start.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);
                    timers
                        .or_ops
                        .fetch_add(1, AtomicOrdering::Relaxed);
                    timers
                        .collect_or_ops
                        .fetch_add(1, AtomicOrdering::Relaxed);
                } else {
                    *edge_entry |= &combined;
                }
                let target_entry = transitions
                    .entry(*lbl)
                    .or_default()
                    .entry(*v)
                    .or_insert_with(Weight::zeros);
                if let Some(timers) = timers {
                    let start = Instant::now();
                    *target_entry |= &combined;
                    timers
                        .or_ns
                        .fetch_add(start.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);
                    timers
                        .or_ops
                        .fetch_add(1, AtomicOrdering::Relaxed);
                    timers
                        .collect_or_ops
                        .fetch_add(1, AtomicOrdering::Relaxed);
                } else {
                    *target_entry |= &combined;
                }
            }
        }
    }
    if let (Some(timers), Some(start)) = (timers, collect_start) {
        timers
            .collect_ns
            .fetch_add(start.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);
    }

    let mut results = Vec::new();
    for (lbl, dest_map) in transitions {
        if let Some(timers) = timers {
            timers.labels.fetch_add(1, AtomicOrdering::Relaxed);
        }
        let Some(w_edge) = edge_weights.remove(&lbl) else {
            continue;
        };
        if w_edge.is_empty() {
            continue;
        }

        let k0 = dest_map.len();

        let expand_start = if timers.is_some() { Some(Instant::now()) } else { None };
        let mut expanded: FxHashMap<NWAStateID, Weight> = FxHashMap::default();
        for (v, w_v) in dest_map {
            if v >= eps_reach.len() {
                continue;
            }
            for (v_reach, w_reach) in &eps_reach[v] {
                let combined = if let Some(timers) = timers {
                    let start = Instant::now();
                    let combined = &w_v & w_reach;
                    timers
                        .and_ns
                        .fetch_add(start.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);
                    timers
                        .and_ops
                        .fetch_add(1, AtomicOrdering::Relaxed);
                    combined
                } else {
                    &w_v & w_reach
                };
                if combined.is_empty() {
                    continue;
                }
                let entry = expanded.entry(*v_reach).or_insert_with(Weight::zeros);
                if !combined.is_subset_of(entry) {
                    if let Some(timers) = timers {
                        let start = Instant::now();
                        *entry |= &combined;
                        timers
                            .or_ns
                            .fetch_add(start.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);
                        timers
                            .or_ops
                            .fetch_add(1, AtomicOrdering::Relaxed);
                        timers
                            .expand_or_ops
                            .fetch_add(1, AtomicOrdering::Relaxed);
                    } else {
                        *entry |= &combined;
                    }
                }
            }
        }
        if let (Some(timers), Some(start)) = (timers, expand_start) {
            timers
                .expand_ns
                .fetch_add(start.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);
        }

        if let Some(timers) = timers {
            if let Ok(mut ratios) = timers.eps_expand_ratios.lock() {
                let k1 = expanded.len();
                let ratio = if k0 == 0 {
                    0.0
                } else {
                    (k1 as f64) / (k0 as f64)
                };
                ratios.push(ratio);
            }
        }

        if expanded.is_empty() {
            continue;
        }

        let normalize_start = if timers.is_some() { Some(Instant::now()) } else { None };
        let w_edge_inv = !&w_edge;
        let mut subset: WeightedSubset = Vec::with_capacity(expanded.len());
        for (sid, w) in expanded {
            let combined = if let Some(timers) = timers {
                let start = Instant::now();
                let combined = w | &w_edge_inv;
                timers
                    .or_ns
                    .fetch_add(start.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);
                timers
                    .or_ops
                    .fetch_add(1, AtomicOrdering::Relaxed);
                timers
                    .normalize_or_ops
                    .fetch_add(1, AtomicOrdering::Relaxed);
                combined
            } else {
                w | &w_edge_inv
            };
            subset.push((sid, combined));
        }
        if let (Some(timers), Some(start)) = (timers, normalize_start) {
            timers
                .normalize_ns
                .fetch_add(start.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);
        }
        results.push((lbl, subset, w_edge));
    }

    results
}


#[derive(Default)]
struct MaterializeTimers {
    collect_ns: AtomicU64,
    expand_ns: AtomicU64,
    normalize_ns: AtomicU64,
    and_ns: AtomicU64,
    or_ns: AtomicU64,
    and_ops: AtomicU64,
    or_ops: AtomicU64,
    collect_or_ops: AtomicU64,
    expand_or_ops: AtomicU64,
    normalize_or_ops: AtomicU64,
    closure_size_total: AtomicU64,
    transition_total: AtomicU64,
    target_total: AtomicU64,
    calls: AtomicU64,
    labels: AtomicU64,
    eps_expand_ratios: Mutex<Vec<f64>>,
    total_updates: AtomicU64,
    closure_lookup_ns: AtomicU64,
    transition_iter_ns: AtomicU64,
    pending_insert_ns: AtomicU64,
}


pub(crate) fn determinize_acyclic_with_progress(
    nwa: &NWA,
    topo_order: &[usize],
    progress_enabled: bool,
) -> DWA {
    if nwa.states.0.is_empty() {
        return DWA::new();
    }

    let macro_level = env::var("MACRO_DEBUG_LEVEL")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(0);
    let profile_enabled = env::var("PROFILE_DETERMINIZATION_BREAKDOWN")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
        || macro_level >= 5;

    let start_time = Instant::now();
    crate::debug!(3, "Determinizing acyclic NWA: precomputing state sets...");

    let max_llm_token = crate::datastructures::get_max_llm_token();
    let num_tsids = crate::datastructures::get_num_tsids();
    crate::datastructures::set_global_dims_all_threads(max_llm_token, num_tsids);

    let eps_unweighted_start = if profile_enabled { Some(Instant::now()) } else { None };
    let eps_reach_unweighted = timeit!("acyclic_det::precompute_eps_closures_unweighted", {
        precompute_all_epsilon_closures_acyclic_unweighted(&nwa.states, topo_order)
    });
    let eps_unweighted_time = eps_unweighted_start.map(|s| s.elapsed());

    let eps_weighted_start = if profile_enabled { Some(Instant::now()) } else { None };
    let eps_reach = timeit!("acyclic_det::precompute_eps_closures", {
        precompute_all_epsilon_closures_acyclic(&nwa.states, topo_order)
    });
    if profile_enabled {
        let mut sizes: Vec<usize> = eps_reach.iter().map(|subset| subset.len()).collect();
        sizes.sort_unstable();
        let len = sizes.len();
        let (avg, p50, p90, p99, max) = if len == 0 {
            (0.0, 0usize, 0usize, 0usize, 0usize)
        } else {
            let sum: usize = sizes.iter().sum();
            let p50_idx = percentile_index(len, 0.50);
            let p90_idx = percentile_index(len, 0.90);
            let p99_idx = percentile_index(len, 0.99);
            (
                (sum as f64) / (len as f64),
                sizes[p50_idx],
                sizes[p90_idx],
                sizes[p99_idx],
                *sizes.last().unwrap(),
            )
        };
        eprintln!(
            "TIMING: determinize_acyclic::eps_closure_sizes avg={:.2} p50={} p90={} p99={} max={}",
            avg,
            p50,
            p90,
            p99,
            max,
        );
    }
    let eps_weighted_time = eps_weighted_start.map(|s| s.elapsed());

    let mut seen_unweighted: FxHashMap<BTreeSet<NWAStateID>, NWAStateID> = FxHashMap::default();
    let mut unweighted_closures: Vec<BTreeSet<NWAStateID>> = Vec::new();
    let mut transition_cache: Vec<Vec<(Label, usize)>> = Vec::new();
    let mut queue: VecDeque<usize> = VecDeque::new();

    fn register_closure_unweighted(
        closure: BTreeSet<NWAStateID>,
        seen: &mut FxHashMap<BTreeSet<NWAStateID>, NWAStateID>,
        closures: &mut Vec<BTreeSet<NWAStateID>>,
        transition_cache: &mut Vec<Vec<(Label, usize)>>,
        queue: &mut VecDeque<usize>,
    ) -> usize {
        if let Some(&id) = seen.get(&closure) {
            return id;
        }
        let id = closures.len();
        seen.insert(closure.clone(), id);
        closures.push(closure);
        transition_cache.push(Vec::new());
        queue.push_back(id);
        id
    }

    let mut start_set: BTreeSet<NWAStateID> = BTreeSet::new();
    for &s in &nwa.body.start_states {
        if s < eps_reach_unweighted.len() {
            start_set.extend(eps_reach_unweighted[s].iter().copied());
        }
    }
    let start_id = timeit!("acyclic_det::register_closure_unweighted", {
        register_closure_unweighted(
            start_set,
            &mut seen_unweighted,
            &mut unweighted_closures,
            &mut transition_cache,
            &mut queue,
        )
    });

    let precompute_start = Instant::now();
    let mut precompute_last_log = Instant::now();
    let precompute_log_interval = Duration::from_secs(2);
    let mut precompute_processed = 0usize;

    let precompute_loop_start = if profile_enabled { Some(Instant::now()) } else { None };
    timeit!("acyclic_det::precompute_loop", {
        while let Some(sid) = queue.pop_front() {
            let closure = unweighted_closures[sid].clone();
            let mut transitions_for_state: Vec<(Label, usize)> = Vec::new();
            for (_lbl, subset) in timeit!("acyclic_det::precompute_build_destinations_unweighted", {
                build_destinations_unweighted(&closure, nwa, &eps_reach_unweighted)
            }) {
                if subset.is_empty() {
                    continue;
                }
                timeit!("acyclic_det::register_closure_unweighted", {
                    let dest_id = register_closure_unweighted(
                        subset,
                        &mut seen_unweighted,
                        &mut unweighted_closures,
                        &mut transition_cache,
                        &mut queue,
                    );
                    transitions_for_state.push((_lbl, dest_id));
                });
            }
            transition_cache[sid] = transitions_for_state;

            precompute_processed += 1;
            if progress_enabled && precompute_last_log.elapsed() >= precompute_log_interval {
                crate::debug!(
                    3,
                    "Determinize precompute: processed={}, discovered={}, queue={}, elapsed={:?}",
                    precompute_processed,
                    unweighted_closures.len(),
                    queue.len(),
                    precompute_start.elapsed(),
                );
                crate::profiler::print_summary();
                crate::profiler::reset();
                precompute_last_log = Instant::now();
            }
        }
    });
    let precompute_loop_time = precompute_loop_start.map(|s| s.elapsed());

    if progress_enabled {
        crate::debug!(
            3,
            "Determinize precompute complete: processed={}, discovered={}, elapsed={:?}",
            precompute_processed,
            unweighted_closures.len(),
            precompute_start.elapsed(),
        );
        crate::profiler::print_summary();
        crate::profiler::reset();
    }

    let total_dwa_states = unweighted_closures.len();
    let total_closure_items: usize = unweighted_closures.iter().map(|c| c.len()).sum();
    let avg_closure_size = if total_dwa_states == 0 {
        0.0
    } else {
        total_closure_items as f64 / total_dwa_states as f64
    };
    let total_transitions: usize = transition_cache.iter().map(|t| t.len()).sum();
    crate::debug!(
        3,
        "Acyclic determinize stats: dwa_states={}, avg_unweighted_closure_size={:.2}, transition_cache_total={}",
        total_dwa_states,
        avg_closure_size,
        total_transitions,
    );

    let mut weighted_closures: Vec<FxHashMap<NWAStateID, Weight>> =
        vec![FxHashMap::default(); unweighted_closures.len()];
    let mut dest_cache: Vec<Vec<(Label, usize, Weight)>> =
        vec![Vec::new(); unweighted_closures.len()];
    let mut start_map: FxHashMap<NWAStateID, Weight> = FxHashMap::default();
    for &s in &nwa.body.start_states {
        if s < eps_reach.len() {
            for (v, w_reach) in &eps_reach[s] {
                start_map
                    .entry(*v)
                    .and_modify(|acc| *acc |= w_reach)
                    .or_insert_with(|| w_reach.clone());
            }
        }
    }
    if start_id < weighted_closures.len() {
        weighted_closures[start_id] = start_map;
    }

    let backend_choice = crate::datastructures::abstract_weight::current_backend_choice();
    let expansion_allowed = crate::datastructures::abstract_weight::is_expansion_allowed();

    let materialize_timers = if profile_enabled {
        Some(Arc::new(MaterializeTimers::default()))
    } else {
        None
    };
    if profile_enabled {
        crate::datastructures::rangemap_weight::set_op_cache_profile_enabled(true);
        crate::datastructures::rangemap_weight::reset_op_cache_or_counters();
        crate::datastructures::hybrid_bitset::set_l1_op_cache_profile_enabled(true);
        crate::datastructures::hybrid_bitset::reset_l1_op_cache_counters();
        crate::datastructures::abstract_weight::reset_bitor_assign_counters();
        crate::datastructures::abstract_weight::reset_bitor_assign_noop_sample();
    }
    let mut materialize_parallel_time: Option<Duration> = None;
    let mut materialize_merge_time: Option<Duration> = None;

    let materialize_start = if profile_enabled { Some(Instant::now()) } else { None };
    timeit!("acyclic_det::materialize_weighted_closures", {
        let dwa_topo = compute_unweighted_dwa_topo_order(&transition_cache);
        let mut levels = vec![0usize; transition_cache.len()];
        for &cid in &dwa_topo {
            let next_level = levels[cid].saturating_add(1);
            for &(_lbl, dest_id) in &transition_cache[cid] {
                if levels[dest_id] < next_level {
                    levels[dest_id] = next_level;
                }
            }
        }
        let max_level = levels.iter().copied().max().unwrap_or(0);
        let mut states_by_level: Vec<Vec<usize>> = vec![Vec::new(); max_level + 1];
        for (cid, level) in levels.into_iter().enumerate() {
            states_by_level[level].push(cid);
        }
        let mut bulk_stats: Option<BTreeMap<usize, (usize, u128)>> = if profile_enabled {
            Some(BTreeMap::new())
        } else {
            None
        };
        let mut merge_or_ops: u64 = 0;
        let mut merge_updates: u64 = 0;
        for states_at_level in states_by_level.into_iter() {
            for &cid in &states_at_level {
                dest_cache[cid].clear();
            }

            let parallel_start = if profile_enabled { Some(Instant::now()) } else { None };
            let updates: Vec<(usize, Vec<(Label, usize, Weight)>, Vec<(usize, WeightedSubset)>)> = {
                let weighted_closures_ref = &weighted_closures;
                let timers = materialize_timers.clone();
                let updates = states_at_level
                    .par_iter()
                    .map_init(
                        || {
                            crate::datastructures::override_backend(backend_choice);
                            crate::datastructures::abstract_weight::override_expansion_allowed(expansion_allowed);
                        },
                        |_, &cid| {
                            let timers = timers.as_deref();
                            let closure_lookup_start = if timers.is_some() {
                                Some(Instant::now())
                            } else {
                                None
                            };
                            let closure = &weighted_closures_ref[cid];
                            let is_empty = closure.is_empty();
                            if let (Some(timers), Some(start)) = (timers, closure_lookup_start) {
                                timers
                                    .closure_lookup_ns
                                    .fetch_add(start.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);
                            }
                            if is_empty {
                                return None;
                            }

                            let cache_len = transition_cache[cid].len();
                            let mut dest_entries: Vec<(Label, usize, Weight)> = Vec::with_capacity(cache_len);
                            let mut weighted_updates: Vec<(usize, WeightedSubset)> = Vec::with_capacity(cache_len);

                            let mut label_to_dest: FxHashMap<Label, usize> = FxHashMap::default();
                            label_to_dest.reserve(cache_len);
                            for &(lbl, dest_id) in &transition_cache[cid] {
                                label_to_dest.insert(lbl, dest_id);
                            }

                            let transition_iter_start = if timers.is_some() {
                                Some(Instant::now())
                            } else {
                                None
                            };
                            for (lbl, dest_subset, w_edge) in
                                build_destinations_batched(closure, nwa, &eps_reach, timers)
                            {
                                let Some(&dest_id) = label_to_dest.get(&lbl) else {
                                    continue;
                                };
                                dest_entries.push((lbl, dest_id, w_edge));
                                weighted_updates.push((dest_id, dest_subset));
                            }
                            if let (Some(timers), Some(start)) = (timers, transition_iter_start) {
                                timers
                                    .transition_iter_ns
                                    .fetch_add(start.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);
                            }

                            Some((cid, dest_entries, weighted_updates))
                        },
                    )
                    .filter_map(|item| item)
                    .collect();
                updates
            };
            if let Some(start) = parallel_start {
                materialize_parallel_time = Some(start.elapsed());
            }

            let merge_start = if profile_enabled { Some(Instant::now()) } else { None };
            let mut pending: FxHashMap<(usize, NWAStateID), Vec<Weight>> = FxHashMap::default();
            let pending_insert_start = if materialize_timers.is_some() {
                Some(Instant::now())
            } else {
                None
            };
            for (cid, dest_entries, weighted_updates) in updates {
                dest_cache[cid] = dest_entries;
                for (dest_id, dest_subset) in weighted_updates {
                    for (sid, w) in dest_subset {
                        if profile_enabled {
                            merge_or_ops = merge_or_ops.saturating_add(1);
                            merge_updates = merge_updates.saturating_add(1);
                        }
                        if let Some(timers) = materialize_timers.as_deref() {
                            timers
                                .total_updates
                                .fetch_add(1, AtomicOrdering::Relaxed);
                        }
                        pending.entry((dest_id, sid)).or_default().push(w);
                    }
                }
            }
            if let (Some(timers), Some(start)) = (materialize_timers.as_deref(), pending_insert_start) {
                timers
                    .pending_insert_ns
                    .fetch_add(start.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);
            }
            for ((dest_id, sid), weights) in pending.iter_mut() {
                if let Some(existing) = weighted_closures[*dest_id].get(sid) {
                    weights.push(existing.clone());
                }
            }
            for ((dest_id, sid), weights) in pending {
                let n = weights.len();
                let mut refs: Vec<&Weight> = Vec::with_capacity(weights.len());
                for w in &weights {
                    refs.push(w);
                }
                let (unioned, elapsed_ns) = if profile_enabled {
                    let start = Instant::now();
                    let unioned = Weight::bulk_union(&refs);
                    (unioned, start.elapsed().as_nanos())
                } else {
                    (Weight::bulk_union(&refs), 0)
                };
                if let Some(stats) = bulk_stats.as_mut() {
                    let entry = stats.entry(n).or_insert((0, 0));
                    entry.0 += 1;
                    entry.1 += elapsed_ns;
                }
                weighted_closures[dest_id].insert(sid, unioned);
            }
            if let Some(start) = merge_start {
                materialize_merge_time = Some(start.elapsed());
            }
        }
        if profile_enabled {
            let final_cells: usize = weighted_closures.iter().map(|m| m.len()).sum();
            let ratio = if final_cells == 0 {
                0.0
            } else {
                (merge_updates as f64) / (final_cells as f64)
            };
            eprintln!(
                "MERGE_STATS: or_ops={} updates={} final_cells={} ratio={:.2}",
                merge_or_ops,
                merge_updates,
                final_cells,
                ratio,
            );
        }
        if let Some(stats) = bulk_stats {
            eprintln!("BULK_UNION_STATS: size -> (count, avg_ns)");
            for (size, (count, total_ns)) in &stats {
                if *count == 0 {
                    continue;
                }
                let avg_ns = total_ns / *count as u128;
                eprintln!(
                    "  n={}: count={}, avg={:.0}ns, total={:.3}ms",
                    size,
                    count,
                    avg_ns,
                    *total_ns as f64 / 1_000_000.0
                );
            }
        }
    });
    let materialize_time = materialize_start.map(|s| s.elapsed());

    let finalize_start = if profile_enabled { Some(Instant::now()) } else { None };
    let closures: Vec<Arc<HashedSubset>> = timeit!("acyclic_det::finalize_weighted_closures", {
        weighted_closures
            .iter()
            .map(|map| {
                let mut subset: WeightedSubset = map.iter().map(|(sid, w)| (*sid, w.clone())).collect();
                subset.sort_unstable_by(|a, b| a.0.cmp(&b.0));
                Arc::new(HashedSubset::from_sorted_vec(subset))
            })
            .collect()
    });
    let finalize_time = finalize_start.map(|s| s.elapsed());

    let mut seen: FxHashMap<Arc<HashedSubset>, NWAStateID> = FxHashMap::default();
    for (id, closure) in closures.iter().enumerate() {
        seen.insert(Arc::clone(closure), id);
    }

    crate::debug!(
        3,
        "Acyclic determinize closures: unweighted={}, weighted={}, unique_weighted={}",
        unweighted_closures.len(),
        closures.len(),
        seen.len(),
    );
    if let Some(start_closure) = closures.get(start_id) {
        let seen_start = seen.get(start_closure).copied().unwrap_or(usize::MAX);
        crate::debug!(
            3,
            "Acyclic determinize start_id={} seen_start_id={}",
            start_id,
            seen_start,
        );
    }

    let total_states = closures.len();
    crate::debug!(3, "Found {} DWA states to process", total_states);

    let processed = AtomicUsize::new(0);
    let next_log = AtomicUsize::new(0);
    let log_every = (total_states / 10).max(1);
    let log_interval = Duration::from_secs(2);
    let last_log = std::sync::Mutex::new(Instant::now());

    let states: Vec<DWAState> = timeit!("acyclic_det::build_states_parallel", {
        closures
            .par_iter()
            .enumerate()
            .map_init(
                || {
                    crate::datastructures::override_backend(backend_choice);
                    crate::datastructures::abstract_weight::override_expansion_allowed(expansion_allowed);
                },
                |_, (idx, closure)| {
                let mut state = DWAState::default();

                let finalw = timeit!("acyclic_det::state_final_weights", {
                    let mut finalw = Weight::zeros();
                    for (sid, cw) in closure.iter() {
                        if let Some(fw) = &nwa.states[*sid].final_weight {
                            let cand = cw & fw;
                            if !cand.is_empty() {
                                finalw |= &cand;
                            }
                        }
                    }
                    finalw
                });
                if !finalw.is_empty() {
                    state.final_weight = Some(finalw);
                }

                let destinations = &dest_cache[idx];
                for (lbl, dest_id, w_edge) in destinations.iter() {
                    let mapped = timeit!("acyclic_det::state_seen_lookup", {
                        seen.get(&closures[*dest_id]).copied()
                    });
                    if let Some(dest_id) = mapped {
                        state.transitions.insert(*lbl, dest_id);
                        state.trans_weights.insert(*lbl, w_edge.clone());
                    }
                }

                if progress_enabled {
                    let count = processed.fetch_add(1, AtomicOrdering::Relaxed) + 1;
                    if count == total_states || count % log_every == 0 {
                        if next_log.load(AtomicOrdering::Relaxed) <= count {
                            let mut last = last_log.lock().unwrap();
                            if last.elapsed() >= log_interval || count == total_states {
                                let pct = (count * 100) / total_states.max(1);
                                crate::debug!(3, "Determinizing: {}% ({}/{})", pct, count, total_states);
                                *last = Instant::now();
                                next_log.store(count + log_every, AtomicOrdering::Relaxed);
                            }
                        }
                    }
                }

                state
            })
            .collect()
    });

    if progress_enabled {
        crate::debug!(
            3,
            "Determinizing: 100% ({}/{}) in {:?}",
            total_states,
            total_states,
            start_time.elapsed(),
        );
        crate::profiler::print_summary();
        crate::profiler::reset();
    }

    if profile_enabled {
        if let Some(t) = eps_unweighted_time {
            eprintln!("TIMING: determinize_acyclic::precompute_eps_unweighted {:?}", t);
        }
        if let Some(t) = eps_weighted_time {
            eprintln!("TIMING: determinize_acyclic::precompute_eps_weighted {:?}", t);
        }
        if let Some(t) = precompute_loop_time {
            eprintln!("TIMING: determinize_acyclic::precompute_loop {:?}", t);
        }
        if let Some(t) = materialize_time {
            eprintln!("TIMING: determinize_acyclic::materialize_weighted_closures {:?}", t);
        }
        if let Some(t) = materialize_parallel_time {
            eprintln!("TIMING: determinize_acyclic::materialize_parallel {:?}", t);
        }
        if let Some(t) = materialize_merge_time {
            eprintln!("TIMING: determinize_acyclic::materialize_merge {:?}", t);
        }
        if let Some(timers) = materialize_timers.as_ref() {
            let collect = Duration::from_nanos(
                timers.collect_ns.load(AtomicOrdering::Relaxed),
            );
            let expand = Duration::from_nanos(
                timers.expand_ns.load(AtomicOrdering::Relaxed),
            );
            let normalize = Duration::from_nanos(
                timers.normalize_ns.load(AtomicOrdering::Relaxed),
            );
            let weight_and = Duration::from_nanos(
                timers.and_ns.load(AtomicOrdering::Relaxed),
            );
            let weight_or = Duration::from_nanos(
                timers.or_ns.load(AtomicOrdering::Relaxed),
            );
            let calls = timers.calls.load(AtomicOrdering::Relaxed);
            let labels = timers.labels.load(AtomicOrdering::Relaxed);
            let and_ops = timers.and_ops.load(AtomicOrdering::Relaxed);
            let or_ops = timers.or_ops.load(AtomicOrdering::Relaxed);
            eprintln!("TIMING: determinize_acyclic::materialize_collect {:?}", collect);
            eprintln!("TIMING: determinize_acyclic::materialize_expand {:?}", expand);
            eprintln!("TIMING: determinize_acyclic::materialize_normalize {:?}", normalize);
            eprintln!(
                "TIMING: determinize_acyclic::materialize_weight_and {:?} ops={}",
                weight_and,
                and_ops,
            );
            eprintln!(
                "TIMING: determinize_acyclic::materialize_weight_or {:?} ops={}",
                weight_or,
                or_ops,
            );
            eprintln!(
                "TIMING: determinize_acyclic::materialize_counts calls={} labels={}",
                calls,
                labels,
            );
            let closure_total = timers
                .closure_size_total
                .load(AtomicOrdering::Relaxed);
            let transition_total = timers
                .transition_total
                .load(AtomicOrdering::Relaxed);
            let target_total = timers
                .target_total
                .load(AtomicOrdering::Relaxed);
            let avg_closure_size = if calls == 0 {
                0.0
            } else {
                (closure_total as f64) / (calls as f64)
            };
            let avg_transitions_per_state = if closure_total == 0 {
                0.0
            } else {
                (transition_total as f64) / (closure_total as f64)
            };
            let avg_targets_per_transition = if transition_total == 0 {
                0.0
            } else {
                (target_total as f64) / (transition_total as f64)
            };
            eprintln!(
                "TIMING: determinize_acyclic::materialize_stats avg_closure_size={:.2} avg_transitions_per_state={:.2} avg_targets_per_transition={:.2}",
                avg_closure_size,
                avg_transitions_per_state,
                avg_targets_per_transition,
            );
            let collect_or = timers
                .collect_or_ops
                .load(AtomicOrdering::Relaxed);
            let expand_or = timers
                .expand_or_ops
                .load(AtomicOrdering::Relaxed);
            let normalize_or = timers
                .normalize_or_ops
                .load(AtomicOrdering::Relaxed);
            let total_or = collect_or
                .saturating_add(expand_or)
                .saturating_add(normalize_or);
            eprintln!(
                "TIMING: determinize_acyclic::materialize_or_ops collect={} expand={} normalize={} total={} all_or_ops={}",
                collect_or,
                expand_or,
                normalize_or,
                total_or,
                or_ops,
            );
            let closure_lookup = Duration::from_nanos(
                timers.closure_lookup_ns.load(AtomicOrdering::Relaxed),
            );
            let transition_iter = Duration::from_nanos(
                timers.transition_iter_ns.load(AtomicOrdering::Relaxed),
            );
            let pending_insert = Duration::from_nanos(
                timers.pending_insert_ns.load(AtomicOrdering::Relaxed),
            );
            eprintln!(
                "TIMING: determinize_acyclic::materialize_blocks closure_lookup={:?} transition_iter={:?} pending_insert={:?}",
                closure_lookup,
                transition_iter,
                pending_insert,
            );
            if let Ok(mut ratios) = timers.eps_expand_ratios.lock() {
                if ratios.is_empty() {
                    eprintln!(
                        "TIMING: determinize_acyclic::eps_expand_ratio avg=0.00 p50=0.00 p90=0.00 p99=0.00 max=0.00",
                    );
                } else {
                    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
                    let len = ratios.len();
                    let sum: f64 = ratios.iter().sum();
                    let avg = sum / (len as f64);
                    let p50 = ratios[percentile_index(len, 0.50)];
                    let p90 = ratios[percentile_index(len, 0.90)];
                    let p99 = ratios[percentile_index(len, 0.99)];
                    let max = *ratios.last().unwrap();
                    eprintln!(
                        "TIMING: determinize_acyclic::eps_expand_ratio avg={:.2} p50={:.2} p90={:.2} p99={:.2} max={:.2}",
                        avg,
                        p50,
                        p90,
                        p99,
                        max,
                    );
                }
            }
            let total_updates = timers.total_updates.load(AtomicOrdering::Relaxed);
            let final_cells: u64 = weighted_closures
                .iter()
                .map(|map| map.len() as u64)
                .sum();
            let update_ratio = if final_cells == 0 {
                0.0
            } else {
                (total_updates as f64) / (final_cells as f64)
            };
            eprintln!(
                "TIMING: determinize_acyclic::weighted_update_ratio total_updates={} final_cells={} ratio={:.2}",
                total_updates,
                final_cells,
                update_ratio,
            );
            let mut total_weights = 0u64;
            let mut all_weights = 0u64;
            let mut empty_weights = 0u64;
            for map in &weighted_closures {
                for weight in map.values() {
                    total_weights += 1;
                    if weight.is_empty() {
                        empty_weights += 1;
                    }
                    if weight.is_all_fast() {
                        all_weights += 1;
                    }
                }
            }
            let all_pct = if total_weights == 0 {
                0.0
            } else {
                (all_weights as f64) * 100.0 / (total_weights as f64)
            };
            let empty_pct = if total_weights == 0 {
                0.0
            } else {
                (empty_weights as f64) * 100.0 / (total_weights as f64)
            };
            eprintln!(
                "TIMING: determinize_acyclic::weight_density total={} all={} ({:.1}%) empty={} ({:.1}%)",
                total_weights,
                all_weights,
                all_pct,
                empty_weights,
                empty_pct,
            );
        }
        {
            let (hits, misses) = crate::datastructures::rangemap_weight::op_cache_or_counters();
            let total = hits.saturating_add(misses);
            let rate = if total == 0 {
                0.0
            } else {
                (hits as f64) * 100.0 / (total as f64)
            };
            eprintln!(
                "TIMING: determinize_acyclic::op_cache_or hits={} misses={} rate={:.1}%",
                hits,
                misses,
                rate,
            );
            let (miss_ns, miss_count) =
                crate::datastructures::rangemap_weight::op_cache_or_miss_time_counters();
            let miss_time = Duration::from_nanos(miss_ns);
            let avg_us = if miss_count == 0 {
                0.0
            } else {
                (miss_ns as f64) / (miss_count as f64) / 1000.0
            };
            eprintln!(
                "TIMING: determinize_acyclic::op_cache_or_miss_time {:?} misses={} avg_us={:.1}",
                miss_time,
                miss_count,
                avg_us,
            );
            let detail = crate::datastructures::rangemap_weight::op_cache_or_miss_detail_counters();
            let prep_time = Duration::from_nanos(detail.prep_ns);
            let union_asym_time = Duration::from_nanos(detail.union_asym_ns);
            let merge_time = Duration::from_nanos(detail.merge_ns);
            let intern_time = Duration::from_nanos(detail.intern_ns);
            let avg_left_ranges = if miss_count == 0 {
                0.0
            } else {
                (detail.left_ranges_total as f64) / (miss_count as f64)
            };
            let avg_right_ranges = if miss_count == 0 {
                0.0
            } else {
                (detail.right_ranges_total as f64) / (miss_count as f64)
            };
            let avg_prep_us = if miss_count == 0 {
                0.0
            } else {
                (detail.prep_ns as f64) / (miss_count as f64) / 1000.0
            };
            let avg_asym_us = if detail.asym_count == 0 {
                0.0
            } else {
                (detail.union_asym_ns as f64) / (detail.asym_count as f64) / 1000.0
            };
            let avg_merge_us = if detail.merge_count == 0 {
                0.0
            } else {
                (detail.merge_ns as f64) / (detail.merge_count as f64) / 1000.0
            };
            let avg_intern_us = if miss_count == 0 {
                0.0
            } else {
                (detail.intern_ns as f64) / (miss_count as f64) / 1000.0
            };
            eprintln!(
                "TIMING: determinize_acyclic::op_cache_or_miss_prep {:?} avg_us={:.1}",
                prep_time,
                avg_prep_us,
            );
            eprintln!(
                "TIMING: determinize_acyclic::op_cache_or_miss_union_asym {:?} count={} avg_us={:.1}",
                union_asym_time,
                detail.asym_count,
                avg_asym_us,
            );
            eprintln!(
                "TIMING: determinize_acyclic::op_cache_or_miss_merge {:?} count={} avg_us={:.1}",
                merge_time,
                detail.merge_count,
                avg_merge_us,
            );
            eprintln!(
                "TIMING: determinize_acyclic::op_cache_or_miss_intern {:?} avg_us={:.1}",
                intern_time,
                avg_intern_us,
            );
            eprintln!(
                "TIMING: determinize_acyclic::op_cache_or_miss_ranges avg_left={:.1} avg_right={:.1}",
                avg_left_ranges,
                avg_right_ranges,
            );
            let rangeset_union_time = Duration::from_nanos(detail.rangeset_union_ns);
            let rangeset_union_avg_us = if detail.rangeset_union_count == 0 {
                0.0
            } else {
                (detail.rangeset_union_ns as f64)
                    / (detail.rangeset_union_count as f64)
                    / 1000.0
            };
            let rangeset_left_avg = if detail.rangeset_union_count == 0 {
                0.0
            } else {
                (detail.rangeset_left_ranges_total as f64)
                    / (detail.rangeset_union_count as f64)
            };
            let rangeset_right_avg = if detail.rangeset_union_count == 0 {
                0.0
            } else {
                (detail.rangeset_right_ranges_total as f64)
                    / (detail.rangeset_union_count as f64)
            };
            eprintln!(
                "TIMING: determinize_acyclic::op_cache_or_miss_rangeset_union {:?} count={} avg_us={:.1}",
                rangeset_union_time,
                detail.rangeset_union_count,
                rangeset_union_avg_us,
            );
            eprintln!(
                "TIMING: determinize_acyclic::op_cache_or_miss_rangeset_sizes avg_left={:.1} avg_right={:.1}",
                rangeset_left_avg,
                rangeset_right_avg,
            );
            let segments_total = detail.segments_total;
            let both = detail.segments_both;
            let left_only = detail.segments_left_only;
            let right_only = detail.segments_right_only;
            let none = detail.segments_none;
            let both_pct = if segments_total == 0 {
                0.0
            } else {
                (both as f64) * 100.0 / (segments_total as f64)
            };
            let left_pct = if segments_total == 0 {
                0.0
            } else {
                (left_only as f64) * 100.0 / (segments_total as f64)
            };
            let right_pct = if segments_total == 0 {
                0.0
            } else {
                (right_only as f64) * 100.0 / (segments_total as f64)
            };
            let none_pct = if segments_total == 0 {
                0.0
            } else {
                (none as f64) * 100.0 / (segments_total as f64)
            };
            eprintln!(
                "TIMING: determinize_acyclic::op_cache_or_miss_segments total={} both={} ({:.1}%) left_only={} ({:.1}%) right_only={} ({:.1}%) none={} ({:.1}%)",
                segments_total,
                both,
                both_pct,
                left_only,
                left_pct,
                right_only,
                right_pct,
                none,
                none_pct,
            );
            let (l1_hits, l1_misses) =
                crate::datastructures::hybrid_bitset::l1_op_cache_or_counters();
            let l1_total = l1_hits.saturating_add(l1_misses);
            let l1_rate = if l1_total == 0 {
                0.0
            } else {
                (l1_hits as f64) * 100.0 / (l1_total as f64)
            };
            eprintln!(
                "TIMING: determinize_acyclic::l1_op_cache_or hits={} misses={} rate={:.1}%",
                l1_hits,
                l1_misses,
                l1_rate,
            );
            let bitor = crate::datastructures::abstract_weight::bitor_assign_counters();
            let total_sample = bitor.union_total;
            let noop_sample = bitor.rhs_empty
                .saturating_add(bitor.self_all)
                .saturating_add(bitor.rhs_all);
            let noop_rate = if total_sample == 0 {
                0.0
            } else {
                (noop_sample as f64) * 100.0 / (total_sample as f64)
            };
            eprintln!(
                "TIMING: determinize_acyclic::bitor_assign_noop total={} noop={} rate={:.1}%",
                total_sample,
                noop_sample,
                noop_rate,
            );
            let (sample_total, sample_noop) =
                crate::datastructures::abstract_weight::bitor_assign_noop_sample_counters();
            let sample_rate = if sample_total == 0 {
                0.0
            } else {
                (sample_noop as f64) * 100.0 / (sample_total as f64)
            };
            eprintln!(
                "TIMING: determinize_acyclic::bitor_assign_noop_sample total={} noop={} rate={:.1}%",
                sample_total,
                sample_noop,
                sample_rate,
            );
        }
        if let Some(t) = finalize_time {
            eprintln!("TIMING: determinize_acyclic::finalize_weighted_closures {:?}", t);
        }
        eprintln!("TIMING: determinize_acyclic::total {:?}", start_time.elapsed());
        eprintln!(
            "TIMING: determinize_acyclic::counters dwa_states={} total_transitions={} avg_unweighted_closure_size={:.2}",
            total_dwa_states,
            total_transitions,
            avg_closure_size,
        );
        crate::datastructures::rangemap_weight::set_op_cache_profile_enabled(false);
        crate::datastructures::hybrid_bitset::set_l1_op_cache_profile_enabled(false);
    }

    DWA {
        states: DWAStates(states),
        body: DWABody { start_state: start_id },
    }
}
