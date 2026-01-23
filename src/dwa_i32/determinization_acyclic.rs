use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::time::{Duration, Instant};

use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use profiler_macro::timeit;

use super::common::{Label, NWAStateID, Weight};
use super::determinization::{HashedSubset, WeightedSubset};
use super::dwa::{DWA, DWABody, DWAState, DWAStates};
use super::nwa::{NWA, NWAStates};

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
    let mut closure_maps: Vec<HashMap<NWAStateID, Weight>> = (0..n)
        .map(|_| HashMap::new())
        .collect();

    for &u in topo.iter().rev() {
        let mut closure: HashMap<NWAStateID, Weight> = HashMap::new();
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

fn build_destination_for_label(
    closure: &HashMap<NWAStateID, Weight>,
    nwa: &NWA,
    eps_reach: &[WeightedSubset],
    label: Label,
) -> Option<(WeightedSubset, Weight)> {
    let mut edge_weight = Weight::zeros();
    let mut dest_map: HashMap<NWAStateID, Weight> = HashMap::new();
    for (u, w_u) in closure {
        let st = &nwa.states[*u];
        let Some(targets) = st.transitions.get(&label) else {
            continue;
        };
        for (v, w_uv) in targets {
            let combined = w_u & w_uv;
            if combined.is_empty() {
                continue;
            }
            edge_weight |= &combined;
            let entry = dest_map.entry(*v).or_insert_with(Weight::zeros);
            *entry |= &combined;
        }
    }
    if edge_weight.is_empty() {
        return None;
    }
    let mut groups: HashMap<NWAStateID, Vec<NWAStateID>> = HashMap::new();
    let mut expanded: HashMap<NWAStateID, Weight> = HashMap::new();
    for (&v, w_v) in dest_map.iter() {
        if v >= eps_reach.len() {
            continue;
        }
        for (v_reach, w_reach) in &eps_reach[v] {
            let is_all = w_reach.is_all_fast();
            if is_all {
                groups.entry(*v_reach).or_default().push(v);
                continue;
            }
            let combined = w_v & w_reach;
            if combined.is_empty() {
                continue;
            }
            let entry = expanded.entry(*v_reach).or_insert_with(Weight::zeros);
            *entry |= &combined;
        }
    }

    for (v_reach, sources) in groups {
        let mut acc = Weight::zeros();
        for v in sources {
            let Some(w_v) = dest_map.get(&v) else {
                continue;
            };
            if w_v.is_empty() {
                continue;
            }
            acc |= w_v;
        }
        if !acc.is_empty() {
            let entry = expanded.entry(v_reach).or_insert_with(Weight::zeros);
            *entry |= &acc;
        }
    }

    if expanded.is_empty() {
        return None;
    }

    let w_edge_inv = !&edge_weight;
    let mut subset: WeightedSubset = Vec::with_capacity(expanded.len());
    for (sid, w) in expanded {
        let merged = w | &w_edge_inv;
        subset.push((sid, merged));
    }
    subset.sort_unstable_by(|a, b| a.0.cmp(&b.0));

    Some((subset, edge_weight))
}


fn build_destinations(
    closure: &HashedSubset,
    nwa: &NWA,
    eps_reach: &[WeightedSubset],
) -> Vec<(Label, WeightedSubset, Weight)> {
    let mut transitions: BTreeMap<Label, HashMap<NWAStateID, Weight>> = BTreeMap::new();
    let mut edge_weights: HashMap<Label, Weight> = HashMap::new();

    timeit!("build_destinations::collect_transitions", {
        for (u, w_u) in closure.iter() {
            timeit!("collect_transitions::closure_item", { () });
            let st = &nwa.states[*u];
            for (lbl, targets) in &st.transitions {
                for (v, w_uv) in targets {
                    timeit!("collect_transitions::transition_item", { () });
                    let combined = timeit!("collect_transitions::weight_and", {
                        w_u & w_uv
                    });
                    if combined.is_empty() {
                        timeit!("collect_transitions::transition_skipped", { () });
                        continue;
                    }
                    timeit!("collect_transitions::transition_kept", { () });
                    timeit!("collect_transitions::edge_weight_update", {
                        let edge_entry = edge_weights.entry(*lbl).or_insert_with(Weight::zeros);
                        *edge_entry |= &combined;
                    });
                    timeit!("collect_transitions::transition_update", {
                        let target_entry = transitions
                            .entry(*lbl)
                            .or_default()
                            .entry(*v)
                            .or_insert_with(Weight::zeros);
                        *target_entry |= &combined;
                    });
                }
            }
        }
    });

    let results = timeit!("build_destinations::expand_and_filter", {
        let mut results = Vec::new();
        for (lbl, dest_map) in transitions {
            let w_edge = match edge_weights.remove(&lbl) {
                Some(w) => w,
                None => continue,
            };
            if w_edge.is_empty() {
                continue;
            }

            let expanded: HashMap<NWAStateID, Weight> = timeit!(
                "build_destinations::eps_reach_expansion",
                {
                    let mut expanded = HashMap::new();
                    for (v, w_v) in dest_map {
                        if v >= eps_reach.len() {
                            continue;
                        }
                        for (v_reach, w_reach) in &eps_reach[v] {
                            let combined = &w_v & w_reach;
                            if combined.is_empty() {
                                continue;
                            }
                            let entry = expanded.entry(*v_reach).or_insert_with(Weight::zeros);
                            if !combined.is_subset_of(entry) {
                                *entry |= &combined;
                            }
                        }
                    }
                    expanded
                }
            );

            if expanded.is_empty() {
                continue;
            }

            let subset: WeightedSubset = timeit!("build_destinations::subset_construction", {
                let w_edge_inv = !&w_edge;
                let mut subset: WeightedSubset = Vec::with_capacity(expanded.len());
                for (sid, w) in expanded {
                    subset.push((sid, w | &w_edge_inv));
                }
                subset.sort_unstable_by(|a, b| a.0.cmp(&b.0));
                subset
            });
            results.push((lbl, subset, w_edge));
        }
        results
    });

    results
}

pub(crate) fn determinize_acyclic_with_progress(
    nwa: &NWA,
    topo_order: &[usize],
    progress_enabled: bool,
) -> DWA {
    if nwa.states.0.is_empty() {
        return DWA::new();
    }

    let start_time = Instant::now();
    crate::debug!(3, "Determinizing acyclic NWA: precomputing state sets...");

    let max_llm_token = crate::datastructures::get_max_llm_token();
    let num_tsids = crate::datastructures::get_num_tsids();
    crate::datastructures::set_global_dims_all_threads(max_llm_token, num_tsids);

    let eps_reach_unweighted = timeit!("acyclic_det::precompute_eps_closures_unweighted", {
        precompute_all_epsilon_closures_acyclic_unweighted(&nwa.states, topo_order)
    });
    let eps_reach = timeit!("acyclic_det::precompute_eps_closures", {
        precompute_all_epsilon_closures_acyclic(&nwa.states, topo_order)
    });

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

    let mut weighted_closures: Vec<HashMap<NWAStateID, Weight>> =
        vec![HashMap::new(); unweighted_closures.len()];
    let mut dest_cache: Vec<Vec<(Label, usize, Weight)>> =
        vec![Vec::new(); unweighted_closures.len()];
    let mut start_map: HashMap<NWAStateID, Weight> = HashMap::new();
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
        for states_at_level in states_by_level.into_iter() {
            for &cid in &states_at_level {
                dest_cache[cid].clear();
            }

            let updates: Vec<(usize, Vec<(Label, usize, Weight)>, Vec<(usize, WeightedSubset)>)> = {
                let weighted_closures_ref = &weighted_closures;
                let updates = states_at_level
                    .par_iter()
                    .map_init(
                        || {
                            crate::datastructures::override_backend(backend_choice);
                            crate::datastructures::abstract_weight::override_expansion_allowed(expansion_allowed);
                        },
                        |_, &cid| {
                            let closure = &weighted_closures_ref[cid];
                            if closure.is_empty() {
                                return None;
                            }

                            let mut dest_entries: Vec<(Label, usize, Weight)> = Vec::new();
                            let mut weighted_updates: Vec<(usize, WeightedSubset)> = Vec::new();
                            for &(lbl, dest_id) in &transition_cache[cid] {
                                let Some((dest_subset, w_edge)) = build_destination_for_label(
                                    closure,
                                    nwa,
                                    &eps_reach,
                                    lbl,
                                ) else {
                                    continue;
                                };

                                dest_entries.push((lbl, dest_id, w_edge));
                                weighted_updates.push((dest_id, dest_subset));
                            }

                            Some((cid, dest_entries, weighted_updates))
                        },
                    )
                    .filter_map(|item| item)
                    .collect();
                updates
            };

            for (cid, dest_entries, weighted_updates) in updates {
                dest_cache[cid] = dest_entries;
                for (dest_id, dest_subset) in weighted_updates {
                    let dest_map = &mut weighted_closures[dest_id];
                    for (sid, w) in dest_subset {
                        let entry = dest_map.entry(sid).or_insert_with(Weight::zeros);
                        *entry |= &w;
                    }
                }
            }
        }
    });
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

    DWA {
        states: DWAStates(states),
        body: DWABody { start_state: start_id },
    }
}
