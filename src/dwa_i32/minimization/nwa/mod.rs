//! NWA minimization passes.

mod prune_unreachable;
mod prune_dead_ends;
mod push_final_weights;
mod push_to_initial;
mod compress;
mod minimize;
mod rebuild;
mod subtract_final_weights;

use super::common::{Partition, MAX_OPTIMIZE_ITERATIONS};
use crate::dwa_i32::common::{Label, NWAStateID, Weight, BENCHMARK_DEBUG};
use crate::dwa_i32::nwa::{NWA, NWABody, NWAState, NWAStates};

use rustfst::algorithms::minimize_with_config;
use rustfst::prelude::MinimizeConfig;

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::time::Instant;
use profiler_macro::{time_it, timeit};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NwaPass {
    PruneUnreachable,
    PruneDeadEnds,
    PushFinalWeights,
    PushWeightsToInitial,
    CompressTransitions,
    Minimize,
    RmEpsilon,
    MinimizeRustfst,  // Full minimize using rustfst
}

impl NwaPass {
    pub fn is_enabled(&self) -> bool {
        match self {
            NwaPass::MinimizeRustfst => {
                std::env::var("NWA_DISABLE_MINIMIZE_RUSTFST")
                    .map(|v| v != "1")
                    .unwrap_or(true)
            }
            _ => true,
        }
    }
}

impl NWA {
    fn try_rm_epsilon_unweighted_scc(&self) -> Option<Vec<NWAState>> {
        let states = &self.states.0;
        let num_states = states.len();
        if num_states == 0 {
            return Some(Vec::new());
        }

        let mut adj: Vec<Vec<NWAStateID>> = vec![Vec::new(); num_states];
        let mut radj: Vec<Vec<NWAStateID>> = vec![Vec::new(); num_states];

        for (u, st) in states.iter().enumerate() {
            for (v, w) in &st.epsilons {
                if !w.is_all_fast() {
                    return None;
                }
                adj[u].push(*v);
                radj[*v].push(u);
            }
        }

        let mut visited = vec![false; num_states];
        let mut order: Vec<usize> = Vec::with_capacity(num_states);
        for start in 0..num_states {
            if visited[start] {
                continue;
            }
            let mut stack: Vec<(usize, usize)> = Vec::new();
            stack.push((start, 0));
            while let Some((node, idx)) = stack.pop() {
                if idx == 0 {
                    if visited[node] {
                        continue;
                    }
                    visited[node] = true;
                }
                if idx < adj[node].len() {
                    stack.push((node, idx + 1));
                    let next = adj[node][idx];
                    if !visited[next] {
                        stack.push((next, 0));
                    }
                } else {
                    order.push(node);
                }
            }
        }

        let mut comp_id = vec![usize::MAX; num_states];
        let mut comp_nodes: Vec<Vec<usize>> = Vec::new();
        for &node in order.iter().rev() {
            if comp_id[node] != usize::MAX {
                continue;
            }
            let cid = comp_nodes.len();
            comp_nodes.push(Vec::new());
            let mut stack: Vec<usize> = vec![node];
            comp_id[node] = cid;
            while let Some(v) = stack.pop() {
                comp_nodes[cid].push(v);
                for &pred in &radj[v] {
                    if comp_id[pred] == usize::MAX {
                        comp_id[pred] = cid;
                        stack.push(pred);
                    }
                }
            }
        }

        let num_comps = comp_nodes.len();
        let mut comp_adj: Vec<Vec<usize>> = vec![Vec::new(); num_comps];
        for u in 0..num_states {
            let cu = comp_id[u];
            for &v in &adj[u] {
                let cv = comp_id[v];
                if cu != cv {
                    comp_adj[cu].push(cv);
                }
            }
        }
        for edges in comp_adj.iter_mut() {
            edges.sort_unstable();
            edges.dedup();
        }

        let mut indegree = vec![0usize; num_comps];
        for c in 0..num_comps {
            for &v in &comp_adj[c] {
                indegree[v] += 1;
            }
        }
        let mut queue: VecDeque<usize> = VecDeque::new();
        for c in 0..num_comps {
            if indegree[c] == 0 {
                queue.push_back(c);
            }
        }
        let mut topo: Vec<usize> = Vec::with_capacity(num_comps);
        while let Some(c) = queue.pop_front() {
            topo.push(c);
            for &v in &comp_adj[c] {
                indegree[v] -= 1;
                if indegree[v] == 0 {
                    queue.push_back(v);
                }
            }
        }
        if topo.len() != num_comps {
            return None;
        }

        let mut comp_transitions: Vec<BTreeMap<Label, HashMap<NWAStateID, Weight>>> =
            vec![BTreeMap::new(); num_comps];
        let mut comp_final: Vec<Option<Weight>> = vec![None; num_comps];

        for (cid, nodes) in comp_nodes.iter().enumerate() {
            let mut trans_map: BTreeMap<Label, HashMap<NWAStateID, Weight>> = BTreeMap::new();
            let mut final_weight: Option<Weight> = None;

            for &u in nodes {
                if let Some(fw) = &states[u].final_weight {
                    if !fw.is_empty() {
                        final_weight = Some(match final_weight {
                            Some(cur) => cur | fw,
                            None => fw.clone(),
                        });
                    }
                }

                for (label, targets) in &states[u].transitions {
                    let entry = trans_map.entry(*label).or_insert_with(HashMap::new);
                    for (tgt, w) in targets {
                        if w.is_empty() {
                            continue;
                        }
                        entry
                            .entry(*tgt)
                            .and_modify(|acc| *acc |= w)
                            .or_insert_with(|| w.clone());
                    }
                }
            }

            comp_transitions[cid] = trans_map;
            comp_final[cid] = final_weight;
        }

        for &c in topo.iter().rev() {
            for &succ in &comp_adj[c] {
                let (c_final, succ_final) = if c < succ {
                    let (left, right) = comp_final.split_at_mut(succ);
                    (&mut left[c], &right[0])
                } else {
                    let (left, right) = comp_final.split_at_mut(c);
                    (&mut right[0], &left[succ])
                };

                if let Some(fw) = succ_final.as_ref() {
                    if !fw.is_empty() {
                        *c_final = Some(match c_final.take() {
                            Some(cur) => cur | fw,
                            None => fw.clone(),
                        });
                    }
                }

                let (c_trans, succ_trans) = if c < succ {
                    let (left, right) = comp_transitions.split_at_mut(succ);
                    (&mut left[c], &right[0])
                } else {
                    let (left, right) = comp_transitions.split_at_mut(c);
                    (&mut right[0], &left[succ])
                };

                for (label, targets) in succ_trans {
                    let entry = c_trans.entry(*label).or_insert_with(HashMap::new);
                    for (tgt, w) in targets {
                        if w.is_empty() {
                            continue;
                        }
                        entry
                            .entry(*tgt)
                            .and_modify(|acc| *acc |= w)
                            .or_insert_with(|| w.clone());
                    }
                }
            }
        }

        let mut new_states: Vec<NWAState> = vec![NWAState::default(); num_states];
        for cid in 0..num_comps {
            let nodes = &comp_nodes[cid];
            let mut new_state = NWAState::default();
            new_state.final_weight = comp_final[cid].clone();
            let trans_map = std::mem::take(&mut comp_transitions[cid]);
            new_state.transitions = trans_map
                .into_iter()
                .map(|(label, map)| (label, map.into_iter().collect()))
                .collect();
            if nodes.len() == 1 {
                new_states[nodes[0]] = new_state;
            } else {
                for &u in nodes {
                    new_states[u] = new_state.clone();
                }
            }
        }

        Some(new_states)
    }

    fn collapse_all_weight_eps(&self) -> Option<NWA> {
        let profile_rm_epsilon = std::env::var("PROFILE_RM_EPSILON").is_ok();
        let mut union_time = std::time::Duration::ZERO;
        let states = &self.states.0;
        let num_states = states.len();
        if num_states == 0 {
            return None;
        }

        let scc_start = Instant::now();
        let mut adj_all: Vec<Vec<NWAStateID>> = vec![Vec::new(); num_states];
        let mut radj_all: Vec<Vec<NWAStateID>> = vec![Vec::new(); num_states];
        let mut all_eps = 0usize;

        for (u, st) in states.iter().enumerate() {
            for (v, w) in &st.epsilons {
                if w.is_all_fast() {
                    adj_all[u].push(*v);
                    radj_all[*v].push(u);
                    all_eps += 1;
                }
            }
        }

        if all_eps == 0 {
            return None;
        }

        let mut visited = vec![false; num_states];
        let mut order: Vec<usize> = Vec::with_capacity(num_states);
        for start in 0..num_states {
            if visited[start] {
                continue;
            }
            let mut stack: Vec<(usize, usize)> = Vec::new();
            stack.push((start, 0));
            while let Some((node, idx)) = stack.pop() {
                if idx == 0 {
                    if visited[node] {
                        continue;
                    }
                    visited[node] = true;
                }
                if idx < adj_all[node].len() {
                    stack.push((node, idx + 1));
                    let next = adj_all[node][idx];
                    if !visited[next] {
                        stack.push((next, 0));
                    }
                } else {
                    order.push(node);
                }
            }
        }

        let mut comp_id = vec![usize::MAX; num_states];
        let mut comp_nodes: Vec<Vec<usize>> = Vec::new();
        for &node in order.iter().rev() {
            if comp_id[node] != usize::MAX {
                continue;
            }
            let cid = comp_nodes.len();
            comp_nodes.push(Vec::new());
            let mut stack: Vec<usize> = vec![node];
            comp_id[node] = cid;
            while let Some(v) = stack.pop() {
                comp_nodes[cid].push(v);
                for &pred in &radj_all[v] {
                    if comp_id[pred] == usize::MAX {
                        comp_id[pred] = cid;
                        stack.push(pred);
                    }
                }
            }
        }

        let num_comps = comp_nodes.len();
        let mut comp_adj: Vec<Vec<usize>> = vec![Vec::new(); num_comps];
        for u in 0..num_states {
            let cu = comp_id[u];
            for &v in &adj_all[u] {
                let cv = comp_id[v];
                if cu != cv {
                    comp_adj[cu].push(cv);
                }
            }
        }
        for edges in comp_adj.iter_mut() {
            edges.sort_unstable();
            edges.dedup();
        }

        let mut indegree = vec![0usize; num_comps];
        for c in 0..num_comps {
            for &v in &comp_adj[c] {
                indegree[v] += 1;
            }
        }
        let mut queue: VecDeque<usize> = VecDeque::new();
        for c in 0..num_comps {
            if indegree[c] == 0 {
                queue.push_back(c);
            }
        }
        let mut topo: Vec<usize> = Vec::with_capacity(num_comps);
        while let Some(c) = queue.pop_front() {
            topo.push(c);
            for &v in &comp_adj[c] {
                indegree[v] -= 1;
                if indegree[v] == 0 {
                    queue.push_back(v);
                }
            }
        }

        let scc_elapsed = scc_start.elapsed();

        let collect_start = Instant::now();

        let mut comp_transitions: Vec<BTreeMap<Label, HashMap<NWAStateID, Weight>>> =
            vec![BTreeMap::new(); num_comps];
        let mut comp_eps: Vec<HashMap<NWAStateID, Weight>> = vec![HashMap::new(); num_comps];
        let mut comp_final: Vec<Option<Weight>> = vec![None; num_comps];

        for (u, st) in states.iter().enumerate() {
            let cu = comp_id[u];
            if let Some(fw) = &st.final_weight {
                if !fw.is_empty() {
                    comp_final[cu] = Some(match comp_final[cu].take() {
                        Some(cur) => {
                            if profile_rm_epsilon {
                                let mut acc = cur;
                                let start = Instant::now();
                                acc |= fw;
                                union_time += start.elapsed();
                                acc
                            } else {
                                cur | fw
                            }
                        }
                        None => fw.clone(),
                    });
                }
            }

            for (label, targets) in &st.transitions {
                let entry = comp_transitions[cu].entry(*label).or_insert_with(HashMap::new);
                for (tgt, w) in targets {
                    if w.is_empty() {
                        continue;
                    }
                    let ct = comp_id[*tgt];
                    entry
                        .entry(ct)
                        .and_modify(|acc| {
                            if profile_rm_epsilon {
                                let start = Instant::now();
                                *acc |= w;
                                union_time += start.elapsed();
                            } else {
                                *acc |= w;
                            }
                        })
                        .or_insert_with(|| w.clone());
                }
            }

            for (v, w) in &st.epsilons {
                if w.is_empty() || w.is_all_fast() {
                    continue;
                }
                let cv = comp_id[*v];
                comp_eps[cu]
                    .entry(cv)
                    .and_modify(|acc| {
                        if profile_rm_epsilon {
                            let start = Instant::now();
                            *acc |= w;
                            union_time += start.elapsed();
                        } else {
                            *acc |= w;
                        }
                    })
                    .or_insert_with(|| w.clone());
            }
        }

        let collect_elapsed = collect_start.elapsed();

        let propagate_start = Instant::now();
        for &c in topo.iter().rev() {
            for &succ in &comp_adj[c] {
                let (c_final, succ_final) = if c < succ {
                    let (left, right) = comp_final.split_at_mut(succ);
                    (&mut left[c], &right[0])
                } else {
                    let (left, right) = comp_final.split_at_mut(c);
                    (&mut right[0], &left[succ])
                };

                if let Some(fw) = succ_final.as_ref() {
                    if !fw.is_empty() {
                        *c_final = Some(match c_final.take() {
                            Some(cur) => cur | fw,
                            None => fw.clone(),
                        });
                    }
                }

                let (c_trans, succ_trans) = if c < succ {
                    let (left, right) = comp_transitions.split_at_mut(succ);
                    (&mut left[c], &right[0])
                } else {
                    let (left, right) = comp_transitions.split_at_mut(c);
                    (&mut right[0], &left[succ])
                };
                for (label, targets) in succ_trans {
                    let entry = c_trans.entry(*label).or_insert_with(HashMap::new);
                    for (tgt, w) in targets {
                        if w.is_empty() {
                            continue;
                        }
                        entry
                            .entry(*tgt)
                            .and_modify(|acc| *acc |= w)
                            .or_insert_with(|| w.clone());
                    }
                }

                let (c_eps, succ_eps) = if c < succ {
                    let (left, right) = comp_eps.split_at_mut(succ);
                    (&mut left[c], &right[0])
                } else {
                    let (left, right) = comp_eps.split_at_mut(c);
                    (&mut right[0], &left[succ])
                };
                for (tgt, w) in succ_eps {
                    if w.is_empty() {
                        continue;
                    }
                    c_eps
                        .entry(*tgt)
                        .and_modify(|acc| {
                            if profile_rm_epsilon {
                                let start = Instant::now();
                                *acc |= w;
                                union_time += start.elapsed();
                            } else {
                                *acc |= w;
                            }
                        })
                        .or_insert_with(|| w.clone());
                }
            }
        }

        let propagate_elapsed = propagate_start.elapsed();

        let build_start = Instant::now();

        let mut new_states: Vec<NWAState> = Vec::with_capacity(num_comps);
        for c in 0..num_comps {
            let mut new_state = NWAState::default();
            new_state.final_weight = comp_final[c].clone();
            let trans_map = std::mem::take(&mut comp_transitions[c]);
            new_state.transitions = trans_map
                .into_iter()
                .map(|(label, map)| (label, map.into_iter().collect()))
                .collect();
            let eps_map = std::mem::take(&mut comp_eps[c]);
            new_state.epsilons = eps_map
                .into_iter()
                .filter(|(_, w)| !w.is_empty())
                .collect();
            new_states.push(new_state);
        }

        let mut new_starts: Vec<NWAStateID> = self
            .body
            .start_states
            .iter()
            .map(|s| comp_id[*s])
            .collect();
        new_starts.sort_unstable();
        new_starts.dedup();

        let build_elapsed = build_start.elapsed();

        if profile_rm_epsilon {
            eprintln!("TIMING: NWA::rm_epsilon::precollapse_scc_elapsed {:?}", scc_elapsed);
            eprintln!("TIMING: NWA::rm_epsilon::precollapse_scc_components {}", num_comps);
            eprintln!("TIMING: NWA::rm_epsilon::precollapse_collect_elapsed {:?}", collect_elapsed);
            eprintln!("TIMING: NWA::rm_epsilon::precollapse_propagate_elapsed {:?}", propagate_elapsed);
            eprintln!("TIMING: NWA::rm_epsilon::precollapse_build_elapsed {:?}", build_elapsed);
            eprintln!("TIMING: NWA::rm_epsilon::precollapse_union {:?}", union_time);
        }

        Some(NWA {
            states: NWAStates(new_states),
            body: NWABody {
                start_states: new_starts,
            },
        })
    }

    fn try_rm_epsilon_weighted_dag_in_place(&mut self) -> bool {
        let num_states = self.states.len();
        if num_states == 0 {
            return true;
        }

        let profile_rm_epsilon = std::env::var("PROFILE_RM_EPSILON").is_ok();
        let dag_start = std::time::Instant::now();
        let mut timing_build_adj = std::time::Duration::ZERO;
        let mut timing_toposort = std::time::Duration::ZERO;
        let mut timing_build_states = std::time::Duration::ZERO;
        let mut total_eps = 0usize;
        let mut states_with_no_eps = 0usize;

        let mut adj: Vec<Vec<(NWAStateID, Weight)>> = vec![Vec::new(); num_states];
        let mut indegree = vec![0usize; num_states];

        let build_adj_start = std::time::Instant::now();
        {
            let states = &self.states.0;
            for (u, st) in states.iter().enumerate() {
                let mut has_eps = false;
                for (v, w) in &st.epsilons {
                    if w.is_empty() {
                        continue;
                    }
                    has_eps = true;
                    total_eps += 1;
                    adj[u].push((*v, w.clone()));
                    indegree[*v] += 1;
                }
                if !has_eps {
                    states_with_no_eps += 1;
                }
            }
        }
        timing_build_adj = build_adj_start.elapsed();

        let toposort_start = std::time::Instant::now();
        let mut queue: VecDeque<NWAStateID> = VecDeque::new();
        for u in 0..num_states {
            if indegree[u] == 0 {
                queue.push_back(u);
            }
        }

        let mut order: Vec<NWAStateID> = Vec::with_capacity(num_states);
        while let Some(u) = queue.pop_front() {
            order.push(u);
            for (v, _) in &adj[u] {
                indegree[*v] -= 1;
                if indegree[*v] == 0 {
                    queue.push_back(*v);
                }
            }
        }
        timing_toposort = toposort_start.elapsed();

        if order.len() != num_states {
            return false; // epsilon graph has cycles
        }

        let build_states_start = std::time::Instant::now();
        for &u in order.iter().rev() {
            if adj[u].is_empty() {
                self.states.0[u].epsilons.clear();
                continue;
            }

            let new_state = {
                let states = &self.states.0;
                let mut final_weight: Option<Weight> = None;
                let mut trans_map: BTreeMap<Label, HashMap<NWAStateID, Weight>> = BTreeMap::new();

                if let Some(fw) = &states[u].final_weight {
                    if !fw.is_empty() {
                        final_weight = Some(fw.clone());
                    }
                }

                for (label, targets) in &states[u].transitions {
                    let entry = trans_map.entry(*label).or_insert_with(HashMap::new);
                    for (tgt, w_tr) in targets {
                        if w_tr.is_empty() {
                            continue;
                        }
                        entry
                            .entry(*tgt)
                            .and_modify(|acc| *acc |= w_tr)
                            .or_insert_with(|| w_tr.clone());
                    }
                }

                for (v, w_uv) in &adj[u] {
                    if w_uv.is_empty() {
                        continue;
                    }
                    let w_uv_all = w_uv.is_all_fast();

                    if let Some(fw) = &states[*v].final_weight {
                        if !fw.is_empty() {
                            let w = if w_uv_all { fw.clone() } else { w_uv & fw };
                            if !w.is_empty() {
                                final_weight = Some(match final_weight {
                                    Some(cur) => cur | &w,
                                    None => w,
                                });
                            }
                        }
                    }

                    for (label, targets) in &states[*v].transitions {
                        let entry = trans_map.entry(*label).or_insert_with(HashMap::new);
                        for (tgt, w_tr) in targets {
                            if w_tr.is_empty() {
                                continue;
                            }
                            let w = if w_uv_all {
                                w_tr.clone()
                            } else if w_tr.is_all_fast() {
                                w_uv.clone()
                            } else {
                                w_uv & w_tr
                            };
                            if w.is_empty() {
                                continue;
                            }
                            entry
                                .entry(*tgt)
                                .and_modify(|acc| *acc |= &w)
                                .or_insert(w);
                        }
                    }
                }

                let mut new_state = NWAState::default();
                new_state.final_weight = final_weight;
                new_state.transitions = trans_map
                    .into_iter()
                    .map(|(label, map)| (label, map.into_iter().collect()))
                    .collect();
                new_state.epsilons.clear();
                new_state
            };

            self.states.0[u] = new_state;
        }
        timing_build_states = build_states_start.elapsed();

        if profile_rm_epsilon {
            let total = dag_start.elapsed();
            let sum = timing_build_adj + timing_toposort + timing_build_states;
            let other = total.saturating_sub(sum);
            eprintln!("TIMING: NWA::rm_epsilon::dag_build_adj {:?}", timing_build_adj);
            eprintln!("TIMING: NWA::rm_epsilon::dag_toposort {:?}", timing_toposort);
            eprintln!("TIMING: NWA::rm_epsilon::dag_build_states {:?}", timing_build_states);
            eprintln!("TIMING: NWA::rm_epsilon::dag_other {:?}", other);
            eprintln!("TIMING: NWA::rm_epsilon::dag_total {:?}", total);
            eprintln!("NWA::rm_epsilon::dag_stats states={} eps={} no_eps_states={}", num_states, total_eps, states_with_no_eps);
        }

        true
    }

    fn try_rm_epsilon_single_source(&mut self, source: NWAStateID) -> bool {
        let profile_rm_epsilon = std::env::var("PROFILE_RM_EPSILON").is_ok();
        let start_time = std::time::Instant::now();
        let mut total_eps = 0usize;

        let new_state = {
            let states = &self.states.0;
            let src_state = &states[source];

            let mut final_weights: Vec<Weight> = Vec::new();
            let mut trans_collected: BTreeMap<Label, HashMap<NWAStateID, Vec<Weight>>> = BTreeMap::new();

            if let Some(fw) = &src_state.final_weight {
                if !fw.is_empty() {
                    final_weights.push(fw.clone());
                }
            }

            for (label, targets) in &src_state.transitions {
                let entry = trans_collected.entry(*label).or_insert_with(HashMap::new);
                for (tgt, w_tr) in targets {
                    if w_tr.is_empty() {
                        continue;
                    }
                    entry.entry(*tgt).or_insert_with(Vec::new).push(w_tr.clone());
                }
            }

            for (tgt, w_eps) in &src_state.epsilons {
                if w_eps.is_empty() {
                    continue;
                }
                total_eps += 1;

                let w_eps_all = w_eps.is_all_fast();
                let tgt_state = &states[*tgt];

                if let Some(fw) = &tgt_state.final_weight {
                    if !fw.is_empty() {
                        let w = if w_eps_all { fw.clone() } else { w_eps & fw };
                        if !w.is_empty() {
                            final_weights.push(w);
                        }
                    }
                }

                for (label, targets) in &tgt_state.transitions {
                    let entry = trans_collected.entry(*label).or_insert_with(HashMap::new);
                    for (next, w_tr) in targets {
                        if w_tr.is_empty() {
                            continue;
                        }
                        let w = if w_eps_all {
                            w_tr.clone()
                        } else if w_tr.is_all_fast() {
                            w_eps.clone()
                        } else {
                            w_eps & w_tr
                        };
                        if w.is_empty() {
                            continue;
                        }
                        entry.entry(*next).or_insert_with(Vec::new).push(w);
                    }
                }
            }

            let final_weight = if final_weights.is_empty() {
                None
            } else if final_weights.len() == 1 {
                Some(final_weights.pop().unwrap())
            } else {
                let refs: Vec<&Weight> = final_weights.iter().collect();
                Some(Weight::bulk_union(&refs))
            };

            let mut transitions: BTreeMap<Label, Vec<(NWAStateID, Weight)>> = BTreeMap::new();
            for (label, targets) in trans_collected {
                let mut merged_targets: Vec<(NWAStateID, Weight)> = Vec::with_capacity(targets.len());
                for (tgt, weights) in targets {
                    let combined = if weights.len() == 1 {
                        weights.into_iter().next().unwrap()
                    } else {
                        let refs: Vec<&Weight> = weights.iter().collect();
                        Weight::bulk_union(&refs)
                    };
                    if combined.is_empty() {
                        continue;
                    }
                    merged_targets.push((tgt, combined));
                }
                if !merged_targets.is_empty() {
                    transitions.insert(label, merged_targets);
                }
            }

            let mut new_state = NWAState::default();
            new_state.final_weight = final_weight;
            new_state.transitions = transitions;
            new_state.epsilons.clear();
            new_state
        };

        self.states.0[source] = new_state;

        if profile_rm_epsilon {
            eprintln!("TIMING: NWA::rm_epsilon::single_source_total {:?}", start_time.elapsed());
            eprintln!("NWA::rm_epsilon::single_source_stats eps={}", total_eps);
        }

        true
    }

    fn rm_epsilon_weighted_in_place(&mut self) {
        if timeit!("NWA::rm_epsilon::dag_weighted", {
            self.try_rm_epsilon_weighted_dag_in_place()
        }) {
            return;
        }

        let profile_rm_epsilon = std::env::var("PROFILE_RM_EPSILON").is_ok();
        let rm_epsilon_start = std::time::Instant::now();
        let mut timing_closure = std::time::Duration::ZERO;
        let mut timing_accumulate = std::time::Duration::ZERO;
        let mut timing_finalize = std::time::Duration::ZERO;
        let mut states_with_no_eps = 0usize;
        let mut total_eps = 0usize;

        let weight_all = Weight::all();
        let states = &self.states.0;
        let num_states = states.len();

        let mut new_states: Vec<NWAState> = vec![NWAState::default(); num_states];
        let mut closure_weights: Vec<Weight> = vec![Weight::zeros(); num_states];
        let mut in_queue: Vec<bool> = vec![false; num_states];
        let mut queue: VecDeque<NWAStateID> = VecDeque::new();
        let mut touched: Vec<NWAStateID> = Vec::new();

        timeit!("NWA::rm_epsilon::build_states", {
            for u in 0..num_states {
                let eps_len = states[u].epsilons.len();
                total_eps += eps_len;
                if eps_len == 0 {
                    states_with_no_eps += 1;
                }

                let closure_start = std::time::Instant::now();
                touched.clear();
                queue.clear();
                closure_weights[u] = weight_all.clone();
                touched.push(u);
                queue.push_back(u);
                in_queue[u] = true;

                while let Some(v) = queue.pop_front() {
                    in_queue[v] = false;
                    let w_uv = closure_weights[v].clone();
                    if w_uv.is_empty() {
                        continue;
                    }
                    for (t, w_vt) in &states[v].epsilons {
                        let new_weight = &w_uv & w_vt;
                        if new_weight.is_empty() {
                            continue;
                        }
                        let updated = &closure_weights[*t] | &new_weight;
                        if updated != closure_weights[*t] {
                            if closure_weights[*t].is_empty() {
                                touched.push(*t);
                            }
                            closure_weights[*t] = updated;
                            if !in_queue[*t] {
                                queue.push_back(*t);
                                in_queue[*t] = true;
                            }
                        }
                    }
                }
                timing_closure += closure_start.elapsed();

                let mut final_weight: Option<Weight> = None;
                let mut trans_map: BTreeMap<Label, HashMap<NWAStateID, Weight>> = BTreeMap::new();

                let accumulate_start = std::time::Instant::now();
                timeit!("NWA::rm_epsilon::accumulate", {
                    for &v in &touched {
                        let w_uv = &closure_weights[v];
                        if w_uv.is_empty() {
                            continue;
                        }
                        let w_uv_all = w_uv.is_all_fast();

                        if let Some(fw) = &states[v].final_weight {
                            if !fw.is_empty() {
                                let w = if w_uv_all { fw.clone() } else { w_uv & fw };
                                if !w.is_empty() {
                                    final_weight = Some(match final_weight {
                                        Some(cur) => cur | &w,
                                        None => w,
                                    });
                                }
                            }
                        }

                        for (label, targets) in &states[v].transitions {
                            let entry = trans_map.entry(*label).or_insert_with(HashMap::new);
                            for (tgt, w_tr) in targets {
                                if w_tr.is_empty() {
                                    continue;
                                }
                                let w = if w_uv_all {
                                    w_tr.clone()
                                } else if w_tr.is_all_fast() {
                                    w_uv.clone()
                                } else {
                                    w_uv & w_tr
                                };
                                if w.is_empty() {
                                    continue;
                                }
                                entry
                                    .entry(*tgt)
                                    .and_modify(|acc| *acc |= &w)
                                    .or_insert(w);
                            }
                        }
                    }
                });
                timing_accumulate += accumulate_start.elapsed();

                let finalize_start = std::time::Instant::now();
                let mut new_state = NWAState::default();
                new_state.final_weight = final_weight;
                new_state.transitions = trans_map
                    .into_iter()
                    .map(|(label, map)| (label, map.into_iter().collect()))
                    .collect();
                new_state.epsilons.clear();
                new_states[u] = new_state;

                for &v in &touched {
                    closure_weights[v] = Weight::zeros();
                    in_queue[v] = false;
                }
                timing_finalize += finalize_start.elapsed();
            }
        });

        self.states.0 = new_states;

        if profile_rm_epsilon {
            let total = rm_epsilon_start.elapsed();
            let sum = timing_closure + timing_accumulate + timing_finalize;
            let other = total.saturating_sub(sum);
            eprintln!("TIMING: NWA::rm_epsilon::closure {:?}", timing_closure);
            eprintln!("TIMING: NWA::rm_epsilon::accumulate {:?}", timing_accumulate);
            eprintln!("TIMING: NWA::rm_epsilon::finalize {:?}", timing_finalize);
            eprintln!("TIMING: NWA::rm_epsilon::other {:?}", other);
            eprintln!("TIMING: NWA::rm_epsilon::total {:?}", total);
            eprintln!("NWA::rm_epsilon::stats states={} eps={} no_eps_states={}", num_states, total_eps, states_with_no_eps);
        }
    }

    fn try_rm_epsilon_single_source_fast(&mut self, profile_rm_epsilon: bool) -> (bool, usize) {
        let mut total_epsilons = 0;
        let mut non_empty_epsilons = 0;
        let mut eps_source: Option<NWAStateID> = None;
        let mut source_count = 0usize;
        let mut has_any_eps = false;
        let mut has_non_empty_eps = false;

        for (u, st) in self.states.0.iter().enumerate() {
            if !st.epsilons.is_empty() {
                has_any_eps = true;
            }
            let mut has_non_empty_eps_for_state = false;
            for (_, w) in &st.epsilons {
                total_epsilons += 1;
                if w.is_empty() {
                    continue;
                }
                non_empty_epsilons += 1;
                has_non_empty_eps_for_state = true;
            }
            if has_non_empty_eps_for_state {
                has_non_empty_eps = true;
                source_count += 1;
                if eps_source.is_none() {
                    eps_source = Some(u);
                }
            }
        }
        let has_multiple_sources = source_count > 1;

        if profile_rm_epsilon {
            eprintln!(
                "NWA::rm_epsilon::single_source_check total_eps={} non_empty_eps={} sources={} has_any_eps={} has_non_empty_eps={} has_multiple_sources={} eps_source={:?}",
                total_epsilons,
                non_empty_epsilons,
                source_count,
                has_any_eps,
                has_non_empty_eps,
                has_multiple_sources,
                eps_source,
            );
        }

        if !has_non_empty_eps {
            if has_any_eps {
                for st in &mut self.states.0 {
                    st.epsilons.clear();
                }
            }
            return (true, total_epsilons);
        }

        if !has_multiple_sources {
            if let Some(src) = eps_source {
                crate::debug!(6, "[NWA] Using single-source epsilon fast path (src={})", src);
                if self.try_rm_epsilon_single_source(src) {
                    for st in &mut self.states.0 {
                        st.epsilons.clear();
                    }
                    return (true, total_epsilons);
                }
            }
        }

        (false, total_epsilons)
    }

    pub fn rm_epsilon(&mut self) {
        crate::debug!(6, "[NWA] Removing epsilon transitions...");
        let profile_rm_epsilon = std::env::var("PROFILE_RM_EPSILON").is_ok();
        let initial_states = self.states.len();
        if initial_states == 0 {
            return;
        }
        let (handled, total_epsilons) = self.try_rm_epsilon_single_source_fast(profile_rm_epsilon);
        crate::debug!(7, "[NWA] Initial number of states: {}, total epsilon transitions: {}", initial_states, total_epsilons);
        if handled {
            return;
        }

        if let Some(new_states) = timeit!("NWA::rm_epsilon::dag_fast_path", {
            self.try_rm_epsilon_unweighted_scc()
        }) {
            self.states.0 = new_states;
            crate::debug!(7, "[NWA] Fast-path epsilon removal applied (all-epsilon SCC)");
            return;
        }

        let use_precollapse = std::env::var("NWA_RM_EPS_PRECOLLAPSE")
            .map(|v| v != "0")
            .unwrap_or(true);
        let mut did_precollapse = false;
        if use_precollapse {
            if profile_rm_epsilon {
                eprintln!("TIMING: NWA::rm_epsilon::precollapse_eps_count {}", total_epsilons);
            }
            let precollapse_start = Instant::now();
            let reduced_opt = self.collapse_all_weight_eps();
            let precollapse_elapsed = precollapse_start.elapsed();
            if profile_rm_epsilon {
                eprintln!("TIMING: NWA::rm_epsilon::precollapse_elapsed {:?}", precollapse_elapsed);
            }
            if let Some(mut reduced) = reduced_opt {
                let before_states = self.states.len();
                let before_eps = total_epsilons;
                let after_states = reduced.states.len();
                let mut after_eps = 0usize;
                for st in &reduced.states.0 {
                    after_eps += st.epsilons.len();
                }
                crate::debug!(
                    6,
                    "[NWA] Pre-collapsed all-weight eps: states {} -> {}, eps {} -> {}",
                    before_states,
                    after_states,
                    before_eps,
                    after_eps,
                );
                let (handled, _) = reduced.try_rm_epsilon_single_source_fast(false);
                if !handled {
                    reduced.rm_epsilon_weighted_in_place();
                }
                *self = reduced;
                did_precollapse = true;
            }
        }

        if !did_precollapse {
            self.rm_epsilon_weighted_in_place();
        }

        let final_states = self.states.len();
        let mut final_epsilons = 0;
        for st in &self.states.0 {
            final_epsilons += st.epsilons.len();
        }
        crate::debug!(7, "[NWA] Final number of states: {}, total epsilon transitions: {}", final_states, final_epsilons);
    }

    pub fn minimize(&mut self) {
        if self.states.len() == 0 {
            return;
        }

        if BENCHMARK_DEBUG {
            let initial_states = self.states.len();
            let mut internal = self.clone();
            let internal_start = std::time::Instant::now();
            internal.minimize_internal();
            let internal_time = internal_start.elapsed();
            let internal_states = internal.states.len();

            let mut rustfst = self.clone();
            let rustfst_start = std::time::Instant::now();
            rustfst.minimize_with_rustfst_full();
            let rustfst_time = rustfst_start.elapsed();
            let rustfst_states = rustfst.states.len();

            if internal_time + rustfst_time > std::time::Duration::from_secs(1) {
                let state_cmp = match internal_states.cmp(&rustfst_states) {
                    std::cmp::Ordering::Less => "<",
                    std::cmp::Ordering::Equal => "=",
                    std::cmp::Ordering::Greater => ">",
                };
                let time_cmp = match internal_time.cmp(&rustfst_time) {
                    std::cmp::Ordering::Less => "<",
                    std::cmp::Ordering::Equal => "=",
                    std::cmp::Ordering::Greater => ">",
                };

                crate::debug!(6, "[NWA Minimize({})] Internal: t={:.2?}, s={} | RustFST: t={:.2?}, s={}. [s: {}, t: {}]", initial_states, internal_time, internal_states, rustfst_time, rustfst_states, state_cmp, time_cmp);
            }

            *self = internal;
        } else {
            self.minimize_internal();
        }
    }

    pub fn minimize_with_rustfst(&mut self) {
        let mut fst = self.to_rustfst();
        minimize_with_config(&mut fst, MinimizeConfig::default().with_allow_nondet(true)).unwrap();
        *self = NWA::from_rustfst(&fst);
    }

    #[time_it("NWA::minimize_with_rustfst_full")]
    pub fn minimize_with_rustfst_full(&mut self) -> bool {
        crate::datastructures::hybrid_bitset::reset_profiling();
        crate::datastructures::rangemap_weight::reset_profiling();
        crate::datastructures::abstract_weight::reset_weight_op_profiling();
        
        let min_config = MinimizeConfig::default().with_allow_nondet(true);
        let (mut fst, to_time) = timeit!("NWA::minimize_rustfst::to_rustfst", {
            let start = Instant::now();
            let fst = self.to_rustfst();
            (fst, start.elapsed())
        });

        let min_time = timeit!("NWA::minimize_rustfst::minimize", {
            let start = Instant::now();
            minimize_with_config(&mut fst, min_config).unwrap();
            start.elapsed()
        });

        let from_time = timeit!("NWA::minimize_rustfst::from_rustfst", {
            let start = Instant::now();
            *self = NWA::from_rustfst(&fst);
            start.elapsed()
        });

        let mut slowest_label = "to_rustfst";
        let mut slowest_time = to_time;
        if min_time > slowest_time {
            slowest_label = "minimize";
            slowest_time = min_time;
        }
        if from_time > slowest_time {
            slowest_label = "from_rustfst";
            slowest_time = from_time;
        }

        crate::debug!(
            4,
            "[NWA::minimize_with_rustfst_full] to_rustfst={:?}, minimize={:?}, from_rustfst={:?}, slowest={} ({:?})",
            to_time,
            min_time,
            from_time,
            slowest_label,
            slowest_time,
        );
        true
    }

    pub fn minimize_internal(&mut self) -> bool {
        crate::debug!(6, "[NWA::minimize] Starting minimization. Initial stats: {}", self.stats());
        let mut total_changed = false;

        let ordering = &[
            NwaPass::PruneUnreachable,
            NwaPass::CompressTransitions,
            NwaPass::PushFinalWeights,
            NwaPass::PushFinalWeights,
            NwaPass::PushWeightsToInitial,
            NwaPass::PruneDeadEnds,
            NwaPass::Minimize,
        ];

        let all_passes: HashSet<NwaPass> = ordering.iter().copied().collect();
        let mut history: Vec<HashSet<NwaPass>> = vec![all_passes.clone(), all_passes];

        let mut force_all_passes = false;
        let mut converged = false;

        for _iter_num in 0..MAX_OPTIMIZE_ITERATIONS {
            let mut current_changing_passes = HashSet::new();
            let mut changed_in_iteration = false;
            for &pass in ordering {
                let recent_activity = history.iter().any(|s| s.contains(&pass));
                if !force_all_passes && !recent_activity && !changed_in_iteration {
                    continue;
                }

                crate::debug!(5, "[NWA::minimize] pass {:?}", pass);
                let pass_changed = match pass {
                    NwaPass::PruneUnreachable => self.prune_unreachable(),
                    NwaPass::PruneDeadEnds => self.prune_dead_ends(),
                    NwaPass::PushFinalWeights => self.push_final_weights_along_epsilons(),
                    NwaPass::PushWeightsToInitial => self.push_weights_to_initial(),
                    NwaPass::CompressTransitions => self.compress_transitions(),
                    NwaPass::Minimize => self.minimize_states(),
                    NwaPass::RmEpsilon => { self.rm_epsilon(); true },
                    NwaPass::MinimizeRustfst => self.minimize_with_rustfst_full(),
                };
                if pass_changed {
                    current_changing_passes.insert(pass);
                }
                changed_in_iteration |= pass_changed;
            }

            history.push(current_changing_passes);
            if history.len() > 2 {
                history.remove(0);
            }

            crate::debug!(5, "[NWA::minimize] iteration done");

            total_changed |= changed_in_iteration;
            if !changed_in_iteration {
                if force_all_passes {
                    converged = true;
                    break;
                }
                force_all_passes = true;
            } else {
                force_all_passes = false;
            }
        }

        if !converged {
            let last_changes = history.last().map(|s| s.iter().copied().collect::<Vec<_>>()).unwrap_or_default();
            crate::debug!(4, "NWA minimization did not converge after {} iterations. Still changing: {:?}", MAX_OPTIMIZE_ITERATIONS, last_changes);
        }

        crate::debug!(6, "[NWA::minimize] Minimization finished. Total changed: {}. Final stats: {}", total_changed, self.stats());
        total_changed
    }
}
