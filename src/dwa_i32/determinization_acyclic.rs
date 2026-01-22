use std::collections::{HashMap, VecDeque};

use super::common::{NWAStateID, Weight};
use super::determinization::WeightedSubset;
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
