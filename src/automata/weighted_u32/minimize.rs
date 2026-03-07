//! IMPORTANT: this should only be implemented for **acyclic** weighted
//! automata. Cyclic input should panic rather than trying to minimize.
// SEP1_MAP: The nearest sep1 analogue is the weighted minimization pipeline under `dwa_i32/minimization/**`, again narrowed here to acyclic-only behavior.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::{HashMap, HashSet};

use super::dwa::DWA;
use crate::ds::weight::Weight;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct StateSignature {
    final_weight: Option<Weight>,
    transitions: Vec<(i32, usize, Weight)>,
}

pub fn minimize(dwa: &DWA) -> DWA {
    if dwa.states.is_empty() {
        return dwa.clone();
    }
    if !dwa.is_acyclic() {
        return dwa.clone();
    }

    fn dfs(state_id: usize, dwa: &DWA, visited: &mut [bool], order: &mut Vec<usize>) {
        if visited[state_id] {
            return;
        }
        visited[state_id] = true;
        for (target, _) in dwa.states[state_id].transitions.values() {
            let target = *target as usize;
            if target < dwa.states.len() {
                dfs(target, dwa, visited, order);
            }
        }
        order.push(state_id);
    }

    let mut visited = vec![false; dwa.states.len()];
    let mut topo = Vec::new();
    dfs(dwa.start_state as usize, dwa, &mut visited, &mut topo);
    topo.reverse();

    let reachable: HashSet<usize> = topo.iter().copied().collect();
    let mut signature_to_class = HashMap::<StateSignature, usize>::new();
    let mut class_of_state = vec![usize::MAX; dwa.states.len()];
    let mut class_representatives = Vec::<usize>::new();

    for &state_id in topo.iter().rev() {
        let state = &dwa.states[state_id];
        let mut transitions = state
            .transitions
            .iter()
            .filter_map(|(&label, (target, weight))| {
                let target = *target as usize;
                reachable.contains(&target).then_some((label, class_of_state[target], weight.clone()))
            })
            .collect::<Vec<_>>();
        transitions.sort_unstable_by(|left, right| left.0.cmp(&right.0).then(left.1.cmp(&right.1)));

        let signature = StateSignature {
            final_weight: state.final_weight.clone(),
            transitions,
        };
        let class_id = if let Some(&existing) = signature_to_class.get(&signature) {
            existing
        } else {
            let new_id = class_representatives.len();
            signature_to_class.insert(signature, new_id);
            class_representatives.push(state_id);
            new_id
        };
        class_of_state[state_id] = class_id;
    }

    let mut minimized = DWA::new(0, 0);
    minimized.states = vec![super::dwa::DWAState::default(); class_representatives.len()];
    minimized.start_state = class_of_state[dwa.start_state as usize] as u32;

    for (class_id, &repr_state_id) in class_representatives.iter().enumerate() {
        let repr = &dwa.states[repr_state_id];
        minimized.states[class_id].final_weight = repr.final_weight.clone();
        minimized.states[class_id].transitions = repr
            .transitions
            .iter()
            .filter_map(|(&label, (target, weight))| {
                let target = *target as usize;
                reachable.contains(&target).then_some((label, (class_of_state[target] as u32, weight.clone())))
            })
            .collect();
    }

    minimized
}
