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
use crate::dwa_i32::nwa::{NWA, NWAState};

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
    pub fn rm_epsilon(&mut self) {
        crate::debug!(6, "[NWA] Removing epsilon transitions...");
        let initial_states = self.states.len();
        if initial_states == 0 {
            return;
        }
        let mut total_epsilons = 0;
        for st in &self.states.0 {
            total_epsilons += st.epsilons.len();
        }
        crate::debug!(7, "[NWA] Initial number of states: {}, total epsilon transitions: {}", initial_states, total_epsilons);

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
                timeit!("NWA::rm_epsilon::closure", {
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
                });

                let mut final_weight: Option<Weight> = None;
                let mut trans_map: BTreeMap<Label, HashMap<NWAStateID, Weight>> = BTreeMap::new();

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
            }
        });

        self.states.0 = new_states;

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
