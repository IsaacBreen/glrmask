use std::collections::VecDeque;
use rustc_hash::FxHashMap;

use super::common::{NWAStateID, Weight};
use super::determinization::WeightedSubset;
use super::nwa::NWAStates;

/// Precomputes the epsilon closure for every state in the NWA.
pub(crate) fn precompute_all_epsilon_closures(states: &NWAStates) -> Vec<WeightedSubset> {
    let n = states.len();
    let mut reachability = Vec::with_capacity(n);

    for start_node in 0..n {
        let mut dists: FxHashMap<NWAStateID, Weight> = FxHashMap::default();
        let mut queue: VecDeque<NWAStateID> = VecDeque::new();

        // Self-reachability is identity
        dists.insert(start_node, Weight::all());
        queue.push_back(start_node);

        while let Some(u) = queue.pop_front() {
            let w_u = dists.get(&u).unwrap().clone();

            if u < n {
                for (v, w_eps) in &states[u].epsilons {
                    let new_w = &w_u & w_eps;
                    if new_w.is_empty() {
                        continue;
                    }

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
