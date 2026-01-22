use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::time::{Duration, Instant};

use rayon::prelude::*;
use rustc_hash::FxHashMap;

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

fn build_destinations(
    closure: &HashedSubset,
    nwa: &NWA,
    eps_reach: &[WeightedSubset],
) -> Vec<(Label, WeightedSubset, Weight)> {
    let mut transitions: BTreeMap<Label, HashMap<NWAStateID, Weight>> = BTreeMap::new();
    let mut edge_weights: HashMap<Label, Weight> = HashMap::new();

    for (u, w_u) in closure.iter() {
        let st = &nwa.states[*u];
        for (lbl, targets) in &st.transitions {
            for (v, w_uv) in targets {
                let combined = w_u & w_uv;
                if combined.is_empty() {
                    continue;
                }
                let edge_entry = edge_weights.entry(*lbl).or_insert_with(Weight::zeros);
                *edge_entry |= &combined;
                let target_entry = transitions
                    .entry(*lbl)
                    .or_default()
                    .entry(*v)
                    .or_insert_with(Weight::zeros);
                *target_entry |= &combined;
            }
        }
    }

    let mut results = Vec::new();
    for (lbl, dest_map) in transitions {
        let w_edge = match edge_weights.remove(&lbl) {
            Some(w) => w,
            None => continue,
        };
        if w_edge.is_empty() {
            continue;
        }

        let mut expanded: HashMap<NWAStateID, Weight> = HashMap::new();
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

        if expanded.is_empty() {
            continue;
        }

        let w_edge_inv = !&w_edge;
        let mut subset: WeightedSubset = Vec::with_capacity(expanded.len());
        for (sid, w) in expanded {
            subset.push((sid, w | &w_edge_inv));
        }
        subset.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        results.push((lbl, subset, w_edge));
    }

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

    let eps_reach = precompute_all_epsilon_closures_acyclic(&nwa.states, topo_order);

    let mut seen: FxHashMap<Arc<HashedSubset>, NWAStateID> = FxHashMap::default();
    let mut closures: Vec<Arc<HashedSubset>> = Vec::new();
    let mut queue: VecDeque<usize> = VecDeque::new();

    fn register_closure(
        hashed: Arc<HashedSubset>,
        seen: &mut FxHashMap<Arc<HashedSubset>, NWAStateID>,
        closures: &mut Vec<Arc<HashedSubset>>,
        queue: &mut VecDeque<usize>,
    ) -> usize {
        if let Some(&id) = seen.get(&hashed) {
            return id;
        }
        let id = closures.len();
        seen.insert(Arc::clone(&hashed), id);
        closures.push(hashed);
        queue.push_back(id);
        id
    }

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
    let mut start_subset: WeightedSubset = start_map.into_iter().collect();
    start_subset.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    let start_id = register_closure(
        Arc::new(HashedSubset::from_sorted_vec(start_subset)),
        &mut seen,
        &mut closures,
        &mut queue,
    );

    let precompute_start = Instant::now();
    let mut precompute_last_log = Instant::now();
    let precompute_log_interval = Duration::from_secs(2);
    let mut precompute_processed = 0usize;

    while let Some(sid) = queue.pop_front() {
        let closure = closures[sid].clone();
        for (_lbl, subset, _w_edge) in build_destinations(&closure, nwa, &eps_reach) {
            if subset.is_empty() {
                continue;
            }
            register_closure(
                Arc::new(HashedSubset::from_sorted_vec(subset)),
                &mut seen,
                &mut closures,
                &mut queue,
            );
        }

        precompute_processed += 1;
        if progress_enabled && precompute_last_log.elapsed() >= precompute_log_interval {
            crate::debug!(
                3,
                "Determinize precompute: processed={}, discovered={}, queue={}, elapsed={:?}",
                precompute_processed,
                closures.len(),
                queue.len(),
                precompute_start.elapsed(),
            );
            precompute_last_log = Instant::now();
        }
    }

    if progress_enabled {
        crate::debug!(
            3,
            "Determinize precompute complete: processed={}, discovered={}, elapsed={:?}",
            precompute_processed,
            closures.len(),
            precompute_start.elapsed(),
        );
    }

    let total_states = closures.len();
    crate::debug!(3, "Found {} DWA states to process", total_states);

    let processed = AtomicUsize::new(0);
    let next_log = AtomicUsize::new(0);
    let log_every = (total_states / 10).max(1);
    let log_interval = Duration::from_secs(2);
    let last_log = std::sync::Mutex::new(Instant::now());

    let backend_choice = crate::datastructures::abstract_weight::current_backend_choice();
    let expansion_allowed = crate::datastructures::abstract_weight::is_expansion_allowed();

    let states: Vec<DWAState> = closures
        .par_iter()
        .map_init(
            || {
                crate::datastructures::override_backend(backend_choice);
                crate::datastructures::abstract_weight::override_expansion_allowed(expansion_allowed);
            },
            |_, closure| {
            let mut state = DWAState::default();

            let mut finalw = Weight::zeros();
            for (sid, cw) in closure.iter() {
                if let Some(fw) = &nwa.states[*sid].final_weight {
                    let cand = cw & fw;
                    if !cand.is_empty() {
                        finalw |= &cand;
                    }
                }
            }
            if !finalw.is_empty() {
                state.final_weight = Some(finalw);
            }

            for (lbl, subset, w_edge) in build_destinations(closure, nwa, &eps_reach) {
                if subset.is_empty() {
                    continue;
                }
                let hashed = HashedSubset::from_sorted_vec(subset);
                if let Some(dest_id) = seen.get(&hashed) {
                    state.transitions.insert(lbl, *dest_id);
                    state.trans_weights.insert(lbl, w_edge);
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
        .collect();

    if progress_enabled {
        crate::debug!(
            3,
            "Determinizing: 100% ({}/{}) in {:?}",
            total_states,
            total_states,
            start_time.elapsed(),
        );
    }

    DWA {
        states: DWAStates(states),
        body: DWABody { start_state: start_id },
    }
}
