//! Resolve negative parser-state labels in weighted NWAs.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::{HashMap, HashSet, VecDeque};

use crate::automata::weighted::nwa::NWA;
use crate::compiler::glr::labels::{DEFAULT_LABEL, is_negative_label, negative_to_positive_label};
use crate::ds::weight::Weight;

type QueryKey = (u32, i32);

#[derive(Clone, Copy)]
enum PredEdge {
    Epsilon { from: usize, eps_idx: usize },
    Negative { from: usize, label: i32, trans_idx: usize },
    Default { from: usize, trans_idx: usize },
}

pub(crate) fn compute_cancellations(nwa: &NWA) -> Vec<(u32, u32, Weight)> {
    let n = nwa.states.len();
    if n == 0 {
        return Vec::new();
    }

    let mut queries: HashMap<u32, HashMap<QueryKey, Weight>> = HashMap::new();
    let mut worklist = VecDeque::<(u32, u32, i32)>::new();
    let mut in_queue = vec![HashSet::<QueryKey>::new(); n];
    let mut new_eps_from: HashMap<u32, HashMap<u32, Weight>> = HashMap::new();

    let merge_into = |entry: &mut Weight, add: Weight| {
        if add.is_empty() {
            return false;
        }
        if entry.is_empty() {
            *entry = add;
            return true;
        }
        let updated = entry.union(&add);
        if updated != *entry {
            *entry = updated;
            true
        } else {
            false
        }
    };

    for a in 0..n {
        for (&label, targets) in &nwa.states[a].transitions {
            if !is_negative_label(label) {
                continue;
            }
            let c = negative_to_positive_label(label);
            for (b, w_ab) in targets {
                if *b as usize >= n || w_ab.is_empty() {
                    continue;
                }
                let query_key = (a as u32, c);
                let entry = queries
                    .entry(*b)
                    .or_default()
                    .entry(query_key)
                    .or_insert_with(Weight::empty);
                if merge_into(entry, w_ab.clone()) {
                    if in_queue[*b as usize].insert(query_key) {
                        worklist.push_back((*b, a as u32, c));
                    }
                }
            }
        }
    }

    while let Some((s, a, c)) = worklist.pop_front() {
        in_queue[s as usize].remove(&(a, c));
        let Some(w_as) = queries.get(&s).and_then(|m| m.get(&(a, c))).cloned() else {
            continue;
        };

        if let Some(epsilons_from_s) = new_eps_from.get(&s) {
            for (&target, eps_w) in epsilons_from_s {
                let prop_w = w_as.intersection(eps_w);
                if prop_w.is_empty() {
                    continue;
                }
                let query_key = (a, c);
                let entry = queries
                    .entry(target)
                    .or_default()
                    .entry(query_key)
                    .or_insert_with(Weight::empty);
                if merge_into(entry, prop_w) {
                    if in_queue[target as usize].insert(query_key) {
                        worklist.push_back((target, a, c));
                    }
                }
            }
        }

        let mut check_cancellations = |target: u32,
                                       w_st: &Weight,
                                       queries: &mut HashMap<u32, HashMap<QueryKey, Weight>>,
                                       worklist: &mut VecDeque<(u32, u32, i32)>,
                                       in_queue: &mut [HashSet<QueryKey>],
                                       new_eps_from: &mut HashMap<u32, HashMap<u32, Weight>>| {
            let new_eps_w = w_as.intersection(w_st);
            if new_eps_w.is_empty() {
                return;
            }

            let eps_entry = new_eps_from
                .entry(a)
                .or_default()
                .entry(target)
                .or_insert_with(Weight::empty);
            let delta_eps = if eps_entry.is_empty() {
                *eps_entry = new_eps_w.clone();
                new_eps_w
            } else {
                let delta_eps = new_eps_w.difference(eps_entry);
                if delta_eps.is_empty() {
                    return;
                }
                *eps_entry = eps_entry.union(&delta_eps);
                delta_eps
            };

            if let Some(queries_at_a) = queries.get(&a).cloned() {
                for ((a_prime, c_prime), w_a_prime_a) in queries_at_a {
                    let prop_w = w_a_prime_a.intersection(&delta_eps);
                    if prop_w.is_empty() {
                        continue;
                    }
                    let query_key = (a_prime, c_prime);
                    let entry = queries
                        .entry(target)
                        .or_default()
                        .entry(query_key)
                        .or_insert_with(Weight::empty);
                    if merge_into(entry, prop_w) {
                        if in_queue[target as usize].insert(query_key) {
                            worklist.push_back((target, a_prime, c_prime));
                        }
                    }
                }
            }
        };

        if let Some(pos_targets) = nwa.states[s as usize].transitions.get(&c) {
            for (t, w_st) in pos_targets {
                if *t as usize >= n {
                    continue;
                }
                check_cancellations(
                    *t,
                    w_st,
                    &mut queries,
                    &mut worklist,
                    &mut in_queue,
                    &mut new_eps_from,
                );
            }
        }

        if let Some(default_targets) = nwa.states[s as usize].transitions.get(&DEFAULT_LABEL) {
            for (target, weight) in default_targets {
                if *target as usize >= n {
                    continue;
                }
                check_cancellations(
                    *target,
                    weight,
                    &mut queries,
                    &mut worklist,
                    &mut in_queue,
                    &mut new_eps_from,
                );
            }
        }

        for (t, w_st) in &nwa.states[s as usize].epsilons {
            if *t as usize >= n {
                continue;
            }
            let prop_w = w_as.intersection(w_st);
            if prop_w.is_empty() {
                continue;
            }
            let query_key = (a, c);
            let entry = queries
                .entry(*t)
                .or_default()
                .entry(query_key)
                .or_insert_with(Weight::empty);
            if merge_into(entry, prop_w) {
                if in_queue[*t as usize].insert(query_key) {
                    worklist.push_back((*t, a, c));
                }
            }
        }
    }

    let mut result = Vec::new();
    for (from, targets) in new_eps_from {
        for (to, w) in targets {
            if !w.is_empty() {
                result.push((from, to, w));
            }
        }
    }

    result
}

pub(crate) fn apply_cancellations(nwa: &mut NWA) {
    for (from, to, weight) in compute_cancellations(nwa) {
        nwa.add_epsilon(from, to, weight);
    }
}

fn merge_final_weight(entry: &mut Option<Weight>, add: Weight) -> bool {
    if add.is_empty() {
        return false;
    }
    match entry {
        Some(existing) => {
            let updated = existing.union(&add);
            if updated != *existing {
                *existing = updated;
                true
            } else {
                false
            }
        }
        None => {
            *entry = Some(add);
            true
        }
    }
}

fn finality_edge_weight<'a>(nwa: &'a NWA, edge: PredEdge) -> (usize, &'a Weight) {
    match edge {
        PredEdge::Epsilon { from, eps_idx } => {
            let (_, weight) = &nwa.states[from].epsilons[eps_idx];
            (from, weight)
        }
        PredEdge::Negative { from, label, trans_idx } => {
            let (_, weight) = &nwa.states[from].transitions[&label][trans_idx];
            (from, weight)
        }
        PredEdge::Default { from, trans_idx } => {
            let (_, weight) = &nwa.states[from].transitions[&DEFAULT_LABEL][trans_idx];
            (from, weight)
        }
    }
}

fn build_finality_predecessors(nwa: &NWA) -> Vec<Vec<PredEdge>> {
    let n = nwa.states.len();
    let mut preds = vec![Vec::<PredEdge>::new(); n];

    for (from, state) in nwa.states.iter().enumerate() {
        for (eps_idx, (target, weight)) in state.epsilons.iter().enumerate() {
            if *target as usize >= n || weight.is_empty() {
                continue;
            }
            preds[*target as usize].push(PredEdge::Epsilon { from, eps_idx });
        }
        for (&label, targets) in &state.transitions {
            if label != DEFAULT_LABEL && !is_negative_label(label) {
                continue;
            }
            for (trans_idx, (target, weight)) in targets.iter().enumerate() {
                if *target as usize >= n || weight.is_empty() {
                    continue;
                }
                if label == DEFAULT_LABEL {
                    preds[*target as usize].push(PredEdge::Default { from, trans_idx });
                } else {
                    preds[*target as usize].push(PredEdge::Negative {
                        from,
                        label,
                        trans_idx,
                    });
                }
            }
        }
    }

    preds
}

fn build_finality_topo_order(nwa: &NWA, graph_profile_enabled: bool) -> Option<Vec<usize>> {
    let n = nwa.states.len();
    let mut indegree = vec![0usize; n];

    for state in &nwa.states {
        for (target, weight) in &state.epsilons {
            if *target as usize >= n || weight.is_empty() {
                continue;
            }
            indegree[*target as usize] += 1;
        }
        for (&label, targets) in &state.transitions {
            if label != DEFAULT_LABEL && !is_negative_label(label) {
                continue;
            }
            for (target, weight) in targets {
                if *target as usize >= n || weight.is_empty() {
                    continue;
                }
                indegree[*target as usize] += 1;
            }
        }
    }

    let mut queue = VecDeque::new();
    for (state_id, degree) in indegree.iter().enumerate() {
        if *degree == 0 {
            queue.push_back(state_id);
        }
    }

    let mut topo_order = Vec::with_capacity(n);
    while let Some(state_id) = queue.pop_front() {
        topo_order.push(state_id);
        let state = &nwa.states[state_id];
        for (target, weight) in &state.epsilons {
            if *target as usize >= n || weight.is_empty() {
                continue;
            }
            indegree[*target as usize] -= 1;
            if indegree[*target as usize] == 0 {
                queue.push_back(*target as usize);
            }
        }
        for (&label, targets) in &state.transitions {
            if label != DEFAULT_LABEL && !is_negative_label(label) {
                continue;
            }
            for (target, weight) in targets {
                if *target as usize >= n || weight.is_empty() {
                    continue;
                }
                indegree[*target as usize] -= 1;
                if indegree[*target as usize] == 0 {
                    queue.push_back(*target as usize);
                }
            }
        }
    }

    let acyclic = topo_order.len() == n;
    if graph_profile_enabled {
        eprintln!(
            "[glrmask/profile][parser_dwa] finality_graph nodes={} acyclic={} cyclic_nodes={}",
            n,
            acyclic,
            n.saturating_sub(topo_order.len()),
        );
    }

    acyclic.then_some(topo_order)
}

fn apply_finality_fixpoint_worklist(
    nwa: &NWA,
    preds: &[Vec<PredEdge>],
    future_final: &mut [Option<Weight>],
) {
    let n = nwa.states.len();
    let mut worklist = VecDeque::<usize>::new();
    let mut in_queue = vec![false; n];

    for state_id in 0..n {
        if future_final[state_id]
            .as_ref()
            .map(|weight| !weight.is_empty())
            .unwrap_or(false)
        {
            in_queue[state_id] = true;
            worklist.push_back(state_id);
        }
    }

    while let Some(state_id) = worklist.pop_front() {
        in_queue[state_id] = false;
        let Some(f_s) = future_final[state_id].clone() else {
            continue;
        };
        if f_s.is_empty() {
            continue;
        }

        for edge in preds[state_id].iter().copied() {
            let (pred_state, edge_weight) = finality_edge_weight(nwa, edge);
            let add = f_s.intersection(edge_weight);
            if merge_final_weight(&mut future_final[pred_state], add) && !in_queue[pred_state] {
                in_queue[pred_state] = true;
                worklist.push_back(pred_state);
            }
        }
    }
}

fn apply_finality_fixpoint_acyclic(
    nwa: &NWA,
    preds: &[Vec<PredEdge>],
    future_final: &mut [Option<Weight>],
    topo_order: &[usize],
) {
    for &state_id in topo_order.iter().rev() {
        let Some(f_s) = future_final[state_id].clone() else {
            continue;
        };
        if f_s.is_empty() {
            continue;
        }

        for edge in preds[state_id].iter().copied() {
            let (pred_state, edge_weight) = finality_edge_weight(nwa, edge);
            let add = f_s.intersection(edge_weight);
            merge_final_weight(&mut future_final[pred_state], add);
        }
    }
}

pub(crate) fn apply_finality_fixpoint(nwa: &mut NWA) {
    let n = nwa.states.len();
    if n == 0 {
        return;
    }
    let graph_profile_enabled = std::env::var_os("GLRMASK_PROFILE_FINALITY_GRAPH").is_some();
    let topo_order = build_finality_topo_order(nwa, graph_profile_enabled);
    let preds = build_finality_predecessors(nwa);

    let mut future_final = vec![None::<Weight>; n];
    for state_id in 0..n {
        if let Some(fw) = nwa.states[state_id].final_weight.clone() {
            if fw.is_empty() {
                continue;
            }
            future_final[state_id] = Some(fw);
        }
    }

    if let Some(topo_order) = topo_order.as_deref() {
        apply_finality_fixpoint_acyclic(nwa, &preds, &mut future_final, topo_order);
    } else {
        apply_finality_fixpoint_worklist(nwa, &preds, &mut future_final);
    }

    for (state_id, final_weight) in future_final.into_iter().enumerate() {
        nwa.states[state_id].final_weight = final_weight.filter(|weight| !weight.is_empty());
    }
}

pub(crate) fn remove_negative_transitions(nwa: &mut NWA) {
    for state in &mut nwa.states {
        state.transitions.retain(|label, _| !is_negative_label(*label));
    }
}

pub(crate) fn remove_redundant_default_transitions(nwa: &mut NWA) {
    let n = nwa.states.len();
    let mut is_terminal = vec![false; n];

    for state_id in 0..n {
        let state = &nwa.states[state_id];
        let has_non_default = state.transitions.iter().any(|(label, targets)| {
            *label != DEFAULT_LABEL && !targets.is_empty()
        });
        let is_final = state.final_weight.as_ref().map(|weight| !weight.is_empty()).unwrap_or(false);
        if !has_non_default && state.epsilons.is_empty() && is_final {
            is_terminal[state_id] = true;
        }
    }

    let mut changed = true;
    while changed {
        changed = false;
        for state_id in 0..n {
            if is_terminal[state_id] {
                continue;
            }
            let state = &nwa.states[state_id];
            let has_non_default = state.transitions.iter().any(|(label, targets)| {
                *label != DEFAULT_LABEL && !targets.is_empty()
            });
            let is_final = state.final_weight.as_ref().map(|weight| !weight.is_empty()).unwrap_or(false);
            if has_non_default || !state.epsilons.is_empty() || !is_final {
                continue;
            }
            let default_targets_terminal = state
                .transitions
                .get(&DEFAULT_LABEL)
                .map(|targets| targets.iter().all(|(target, _)| is_terminal[*target as usize]))
                .unwrap_or(true);
            if default_targets_terminal {
                is_terminal[state_id] = true;
                changed = true;
            }
        }
    }

    for state in &mut nwa.states {
        if let Some(targets) = state.transitions.get_mut(&DEFAULT_LABEL) {
            targets.retain(|(target, _)| !is_terminal[*target as usize]);
        }
        state.transitions.retain(|_, targets| !targets.is_empty());
    }
}

pub(crate) fn resolve_negative_codes_in_nwa(nwa: &mut NWA) {
    let profile_enabled = std::env::var_os("GLRMASK_PROFILE_PARSER_DWA").is_some();

    let phase_started_at = std::time::Instant::now();
    apply_cancellations(nwa);
    let apply_cancellations_time = phase_started_at.elapsed();

    let phase_started_at = std::time::Instant::now();
    apply_finality_fixpoint(nwa);
    let apply_finality_fixpoint_time = phase_started_at.elapsed();

    let phase_started_at = std::time::Instant::now();
    remove_negative_transitions(nwa);
    let remove_negative_transitions_time = phase_started_at.elapsed();

    let phase_started_at = std::time::Instant::now();
    remove_redundant_default_transitions(nwa);
    let remove_redundant_default_transitions_time = phase_started_at.elapsed();

    if profile_enabled {
        eprintln!(
            "[glrmask/profile][parser_dwa] resolve_negative_codes_detail apply_cancellations_ms={:.3} apply_finality_fixpoint_ms={:.3} remove_negative_transitions_ms={:.3} remove_redundant_default_transitions_ms={:.3}",
            apply_cancellations_time.as_secs_f64() * 1000.0,
            apply_finality_fixpoint_time.as_secs_f64() * 1000.0,
            remove_negative_transitions_time.as_secs_f64() * 1000.0,
            remove_redundant_default_transitions_time.as_secs_f64() * 1000.0,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::glr::labels::{encode_negative_label, encode_positive_label};
    use std::collections::BTreeMap;

    fn compute_cancellations_reference(nwa: &NWA) -> Vec<(u32, u32, Weight)> {
        let n = nwa.states.len();
        if n == 0 {
            return Vec::new();
        }

        let mut queries: HashMap<u32, HashMap<QueryKey, Weight>> = HashMap::new();
        let mut worklist = VecDeque::<(u32, u32, i32)>::new();
        let mut in_queue = vec![HashSet::<QueryKey>::new(); n];
        let mut new_eps_from: HashMap<u32, HashMap<u32, Weight>> = HashMap::new();

        let merge_into = |entry: &mut Weight, add: Weight| {
            if add.is_empty() {
                return false;
            }
            if entry.is_empty() {
                *entry = add;
                return true;
            }
            let updated = entry.union(&add);
            if updated != *entry {
                *entry = updated;
                true
            } else {
                false
            }
        };

        for a in 0..n {
            for (&label, targets) in &nwa.states[a].transitions {
                if !is_negative_label(label) {
                    continue;
                }
                let c = negative_to_positive_label(label);
                for (b, w_ab) in targets {
                    if *b as usize >= n || w_ab.is_empty() {
                        continue;
                    }
                    let query_key = (a as u32, c);
                    let entry = queries
                        .entry(*b)
                        .or_default()
                        .entry(query_key)
                        .or_insert_with(Weight::empty);
                    if merge_into(entry, w_ab.clone()) {
                        if in_queue[*b as usize].insert(query_key) {
                            worklist.push_back((*b, a as u32, c));
                        }
                    }
                }
            }
        }

        while let Some((s, a, c)) = worklist.pop_front() {
            in_queue[s as usize].remove(&(a, c));
            let Some(w_as) = queries.get(&s).and_then(|m| m.get(&(a, c))).cloned() else {
                continue;
            };

            if let Some(epsilons_from_s) = new_eps_from.get(&s) {
                for (&target, eps_w) in epsilons_from_s {
                    let prop_w = w_as.intersection(eps_w);
                    if prop_w.is_empty() {
                        continue;
                    }
                    let query_key = (a, c);
                    let entry = queries
                        .entry(target)
                        .or_default()
                        .entry(query_key)
                        .or_insert_with(Weight::empty);
                    if merge_into(entry, prop_w) {
                        if in_queue[target as usize].insert(query_key) {
                            worklist.push_back((target, a, c));
                        }
                    }
                }
            }

            let mut check_cancellations = |target: u32,
                                           w_st: &Weight,
                                           queries: &mut HashMap<u32, HashMap<QueryKey, Weight>>,
                                           worklist: &mut VecDeque<(u32, u32, i32)>,
                                           in_queue: &mut [HashSet<QueryKey>],
                                           new_eps_from: &mut HashMap<u32, HashMap<u32, Weight>>| {
                let new_eps_w = w_as.intersection(w_st);
                if new_eps_w.is_empty() {
                    return;
                }

                let eps_entry = new_eps_from
                    .entry(a)
                    .or_default()
                    .entry(target)
                    .or_insert_with(Weight::empty);
                let updated_eps = if eps_entry.is_empty() {
                    new_eps_w
                } else {
                    eps_entry.union(&new_eps_w)
                };
                if updated_eps == *eps_entry {
                    return;
                }
                *eps_entry = updated_eps.clone();

                if let Some(queries_at_a) = queries.get(&a).cloned() {
                    for ((a_prime, c_prime), w_a_prime_a) in queries_at_a {
                        let prop_w = w_a_prime_a.intersection(&updated_eps);
                        if prop_w.is_empty() {
                            continue;
                        }
                        let query_key = (a_prime, c_prime);
                        let entry = queries
                            .entry(target)
                            .or_default()
                            .entry(query_key)
                            .or_insert_with(Weight::empty);
                        if merge_into(entry, prop_w) {
                            if in_queue[target as usize].insert(query_key) {
                                worklist.push_back((target, a_prime, c_prime));
                            }
                        }
                    }
                }
            };

            if let Some(pos_targets) = nwa.states[s as usize].transitions.get(&c) {
                for (t, w_st) in pos_targets {
                    if *t as usize >= n {
                        continue;
                    }
                    check_cancellations(
                        *t,
                        w_st,
                        &mut queries,
                        &mut worklist,
                        &mut in_queue,
                        &mut new_eps_from,
                    );
                }
            }

            if let Some(default_targets) = nwa.states[s as usize].transitions.get(&DEFAULT_LABEL) {
                for (target, weight) in default_targets {
                    if *target as usize >= n {
                        continue;
                    }
                    check_cancellations(
                        *target,
                        weight,
                        &mut queries,
                        &mut worklist,
                        &mut in_queue,
                        &mut new_eps_from,
                    );
                }
            }

            for (t, w_st) in &nwa.states[s as usize].epsilons {
                if *t as usize >= n {
                    continue;
                }
                let prop_w = w_as.intersection(w_st);
                if prop_w.is_empty() {
                    continue;
                }
                let query_key = (a, c);
                let entry = queries
                    .entry(*t)
                    .or_default()
                    .entry(query_key)
                    .or_insert_with(Weight::empty);
                if merge_into(entry, prop_w) {
                    if in_queue[*t as usize].insert(query_key) {
                        worklist.push_back((*t, a, c));
                    }
                }
            }
        }

        let mut result = Vec::new();
        for (from, targets) in new_eps_from {
            for (to, w) in targets {
                if !w.is_empty() {
                    result.push((from, to, w));
                }
            }
        }

        result
    }

    fn normalize_cancellations(cancellations: Vec<(u32, u32, Weight)>) -> BTreeMap<(u32, u32), Weight> {
        cancellations
            .into_iter()
            .map(|(from, to, weight)| ((from, to), weight))
            .collect()
    }

    fn add_weighted_transition(nwa: &mut NWA, from: u32, label: i32, to: u32, kind: u8) {
        match kind {
            1 => nwa.add_transition(from, label, to, weight_1()),
            2 => nwa.add_transition(from, label, to, weight_12()),
            _ => {}
        }
    }

    fn add_weighted_epsilon(nwa: &mut NWA, from: u32, to: u32, kind: u8) {
        match kind {
            1 => nwa.add_epsilon(from, to, weight_1()),
            2 => nwa.add_epsilon(from, to, weight_12()),
            _ => {}
        }
    }

    fn set_weighted_final(nwa: &mut NWA, state: u32, kind: u8) {
        match kind {
            1 => nwa.set_final_weight(state, weight_1()),
            2 => nwa.set_final_weight(state, weight_12()),
            _ => {}
        }
    }

    fn apply_finality_fixpoint_reference(nwa: &mut NWA) {
        let n = nwa.states.len();
        if n == 0 {
            return;
        }

        let mut preds = vec![Vec::<PredEdge>::new(); n];
        for (from, state) in nwa.states.iter().enumerate() {
            for (eps_idx, (target, weight)) in state.epsilons.iter().enumerate() {
                if *target as usize >= n || weight.is_empty() {
                    continue;
                }
                preds[*target as usize].push(PredEdge::Epsilon { from, eps_idx });
            }
            for (&label, targets) in &state.transitions {
                if label != DEFAULT_LABEL && !is_negative_label(label) {
                    continue;
                }
                for (trans_idx, (target, weight)) in targets.iter().enumerate() {
                    if *target as usize >= n || weight.is_empty() {
                        continue;
                    }
                    if label == DEFAULT_LABEL {
                        preds[*target as usize].push(PredEdge::Default { from, trans_idx });
                    } else {
                        preds[*target as usize].push(PredEdge::Negative {
                            from,
                            label,
                            trans_idx,
                        });
                    }
                }
            }
        }

        let mut future_final = vec![None::<Weight>; n];
        for state_id in 0..n {
            if let Some(fw) = nwa.states[state_id].final_weight.clone() {
                if fw.is_empty() {
                    continue;
                }
                future_final[state_id] = Some(fw);
            }
        }

        apply_finality_fixpoint_worklist(nwa, &preds, &mut future_final);

        for (state_id, final_weight) in future_final.into_iter().enumerate() {
            nwa.states[state_id].final_weight = final_weight.filter(|weight| !weight.is_empty());
        }
    }

    fn final_weights(nwa: &NWA) -> Vec<Option<Weight>> {
        nwa.states.iter().map(|state| state.final_weight.clone()).collect()
    }

    fn weight_1() -> Weight {
        Weight::from_compact_ranges([(1..=1, [1..=1])])
    }

    fn weight_12() -> Weight {
        Weight::from_compact_ranges([(1..=2, [1..=2])])
    }

    #[test]
    fn test_compute_cancellations_widens_existing_eps_query() {
        let mut nwa = NWA::new(0, 0);
        for _ in 0..3 {
            nwa.add_state();
        }

        let neg1 = encode_negative_label(1);
        let pos1 = encode_positive_label(1);

        nwa.add_transition(0, neg1, 0, weight_1());
        nwa.add_transition(0, neg1, 2, weight_12());
        nwa.add_epsilon(0, 1, weight_1());
        nwa.add_transition(1, pos1, 2, weight_12());
        nwa.add_transition(2, pos1, 1, weight_12());
        nwa.add_epsilon(2, 0, weight_12());

        let cancellations = compute_cancellations(&nwa);

        assert!(
            cancellations
                .iter()
                .any(|(from, to, w)| *from == 0 && *to == 2 && *w == weight_12()),
            "expected widened epsilon 0->2 with weight_12"
        );
        assert!(
            !cancellations
                .iter()
                .any(|(from, to, w)| *from == 0 && *to == 2 && *w == weight_1()),
            "narrower 0->2 weight_1 should have been widened away"
        );
    }

    #[test]
    fn test_compute_cancellations_propagates_later_query_through_existing_epsilon() {
        let mut nwa = NWA::new(0, 0);
        for _ in 0..4 {
            nwa.add_state();
        }

        let neg1 = encode_negative_label(1);
        let pos1 = encode_positive_label(1);

        nwa.add_transition(0, neg1, 1, weight_12());
        nwa.add_transition(1, pos1, 2, weight_12());
        nwa.add_transition(3, neg1, 0, weight_1());
        nwa.add_transition(2, pos1, 1, weight_12());

        let cancellations = compute_cancellations(&nwa);

        assert!(
            cancellations
                .iter()
                .any(|(from, to, w)| *from == 0 && *to == 2 && *w == weight_12()),
            "expected initial epsilon 0->2 with weight_12"
        );
        assert!(
            cancellations
                .iter()
                .any(|(from, to, w)| *from == 3 && *to == 1 && *w == weight_1()),
            "expected later query to reuse existing epsilon and create 3->1 with weight_1"
        );
    }

    #[test]
    fn test_compute_cancellations_delta_matches_reference_on_small_family() {
        let neg1 = encode_negative_label(1);
        let pos1 = encode_positive_label(1);

        for config in 0..(3usize.pow(7)) {
            let mut code = config;
            let mut next_kind = || {
                let kind = (code % 3) as u8;
                code /= 3;
                kind
            };

            let mut nwa = NWA::new(0, 0);
            for _ in 0..4 {
                nwa.add_state();
            }

            add_weighted_transition(&mut nwa, 0, neg1, 0, next_kind());
            add_weighted_transition(&mut nwa, 0, neg1, 2, next_kind());
            add_weighted_epsilon(&mut nwa, 0, 1, next_kind());
            add_weighted_transition(&mut nwa, 1, pos1, 2, next_kind());
            add_weighted_transition(&mut nwa, 2, pos1, 1, next_kind());
            add_weighted_epsilon(&mut nwa, 2, 0, next_kind());
            add_weighted_transition(&mut nwa, 3, neg1, 0, next_kind());

            let actual = normalize_cancellations(compute_cancellations(&nwa));
            let expected = normalize_cancellations(compute_cancellations_reference(&nwa));

            assert_eq!(
                actual, expected,
                "delta propagation diverged from reference for config {}",
                config
            );
        }
    }

    #[test]
    fn test_apply_finality_fixpoint_matches_reference_on_small_acyclic_family() {
        let neg1 = encode_negative_label(1);

        for config in 0..(3usize.pow(6)) {
            let mut code = config;
            let mut next_kind = || {
                let kind = (code % 3) as u8;
                code /= 3;
                kind
            };

            let mut nwa = NWA::new(0, 0);
            for _ in 0..3 {
                nwa.add_state();
            }

            set_weighted_final(&mut nwa, 0, next_kind());
            set_weighted_final(&mut nwa, 1, next_kind());
            set_weighted_final(&mut nwa, 2, next_kind());
            add_weighted_epsilon(&mut nwa, 0, 1, next_kind());
            add_weighted_transition(&mut nwa, 0, DEFAULT_LABEL, 2, next_kind());
            add_weighted_transition(&mut nwa, 1, neg1, 2, next_kind());

            let mut actual = nwa.clone();
            let mut expected = nwa.clone();
            apply_finality_fixpoint(&mut actual);
            apply_finality_fixpoint_reference(&mut expected);

            assert_eq!(
                final_weights(&actual),
                final_weights(&expected),
                "acyclic finality propagation diverged for config {}",
                config
            );
        }
    }

    #[test]
    fn test_apply_finality_fixpoint_matches_reference_on_small_cyclic_family() {
        let neg1 = encode_negative_label(1);

        for config in 0..(3usize.pow(7)) {
            let mut code = config;
            let mut next_kind = || {
                let kind = (code % 3) as u8;
                code /= 3;
                kind
            };

            let mut nwa = NWA::new(0, 0);
            for _ in 0..3 {
                nwa.add_state();
            }

            set_weighted_final(&mut nwa, 0, next_kind());
            set_weighted_final(&mut nwa, 1, next_kind());
            set_weighted_final(&mut nwa, 2, next_kind());
            add_weighted_epsilon(&mut nwa, 0, 1, next_kind());
            add_weighted_transition(&mut nwa, 1, DEFAULT_LABEL, 0, next_kind());
            add_weighted_transition(&mut nwa, 1, neg1, 2, next_kind());
            add_weighted_epsilon(&mut nwa, 2, 1, next_kind());

            let mut actual = nwa.clone();
            let mut expected = nwa.clone();
            apply_finality_fixpoint(&mut actual);
            apply_finality_fixpoint_reference(&mut expected);

            assert_eq!(
                final_weights(&actual),
                final_weights(&expected),
                "cyclic finality propagation diverged for config {}",
                config
            );
        }
    }
}
