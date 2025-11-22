#![allow(dead_code)]
#![allow(clippy::needless_borrow)]

use chrono::Local;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, VecDeque};

use super::common::{DETERMINIZE_DEBUG, Label, NWAStateID, Weight};
use super::dwa::DWA;
use super::nwa::{NWA, NWAStates};
use crate::precompute4::test_weighted_automata;

// Invariants: strictly sorted by NWAStateID, no duplicate IDs, no empty Weights.
type WeightedSubset = Vec<(NWAStateID, Weight)>;

fn is_zero(w: &Weight) -> bool { w.is_empty() }

// Optimized epsilon closure with correct change detection.
// Uses HashMap for fast lookups during accumulation, returns a sorted Vec.
fn epsilon_closure_optimized(nwa_states: &NWAStates, seed: &WeightedSubset) -> WeightedSubset {
    let mut closure_map: HashMap<NWAStateID, Weight> = HashMap::with_capacity(seed.len() * 4);
    let mut queue: VecDeque<NWAStateID> = VecDeque::with_capacity(seed.len());

    for (sid, w) in seed {
        if !is_zero(w) {
            closure_map.insert(*sid, w.clone());
            queue.push_back(*sid);
        }
    }

    while let Some(u) = queue.pop_front() {
        // Snapshot the weight of u to propagate
        let uw = if let Some(w) = closure_map.get(&u) {
            w.clone()
        } else {
            continue;
        };

        if u >= nwa_states.len() { continue; }
        let st = &nwa_states[u];

        for (v, w_eps) in &st.epsilons {
            let cand = &uw & w_eps;
            if cand.is_empty() { continue; }

            let entry = closure_map.entry(*v).or_insert_with(Weight::zeros);

            // If the candidate weight adds new information (is not a subset of existing),
            // update and propagate.
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

struct Determinizer<'a> {
    nwa: &'a NWA,
    // Map from canonical closure (Sorted Vec) to DWA State ID
    seen: HashMap<WeightedSubset, usize>,
    // Queue of DWA State IDs to process
    queue: VecDeque<usize>,
    // Store the closure for each DWA state to avoid recomputing/cloning implicitly
    closures: Vec<WeightedSubset>,
    dwa: DWA,
    mp: Option<MultiProgress>,
}

impl<'a> Determinizer<'a> {
    fn new(nwa: &'a NWA, mp: Option<MultiProgress>) -> Self {
        let mut dwa = DWA::new();
        dwa.states.0.clear();
        dwa.body.start_state = 0;
        Determinizer {
            nwa,
            seen: HashMap::new(),
            queue: VecDeque::new(),
            closures: Vec::new(),
            dwa,
            mp,
        }
    }

    // Returns the DWA state ID for the given closure. If new, registers it.
    fn register_closure(&mut self, closure: WeightedSubset) -> usize {
        if let Some(&id) = self.seen.get(&closure) {
            return id;
        }

        let id = self.dwa.add_state();

        // Compute final weights
        let mut finalw = Weight::zeros();
        for (sid, cw) in &closure {
            if let Some(fw) = &self.nwa.states[*sid].final_weight {
                let cand = cw & fw;
                if !cand.is_empty() {
                    finalw |= &cand;
                }
            }
        }
        if !finalw.is_empty() {
            let _ = self.dwa.set_final_weight(id, finalw);
        }

        self.seen.insert(closure.clone(), id);
        self.closures.push(closure);
        self.queue.push_back(id);
        id
    }

    fn expand_state(&mut self, sid: usize) {
        let closure = self.closures[sid].clone();

        if closure.is_empty() {
            return;
        }

        // 1. Collect raw outgoing transitions: (Label, TargetNWAState, Weight)
        // Capacity guess: closure size * small factor
        let mut raw: Vec<(Label, NWAStateID, Weight)> = Vec::with_capacity(closure.len() * 2);

        for (u, w_u) in &closure {
            let st = &self.nwa.states[*u];
            for (lbl, targets) in &st.transitions {
                for (v, w_trans) in targets {
                    let w_out = w_u & w_trans;
                    if !w_out.is_empty() {
                        raw.push((*lbl, *v, w_out));
                    }
                }
            }
        }

        if raw.is_empty() {
            return;
        }

        // 2. Sort by Label, then Target.
        // This groups transitions for the same label and same target together.
        raw.sort_unstable_by(|a, b| {
            let c = a.0.cmp(&b.0);
            if c != Ordering::Equal {
                return c;
            }
            a.1.cmp(&b.1)
        });

        // 3. Iterate through sorted transitions to build next subsets.
        let mut i = 0;
        while i < raw.len() {
            let current_lbl = raw[i].0;

            // Prepare to build the target subset for this label.
            // Since `raw` is sorted by target within label, we can collect sequentially.
            let mut next_subset_raw: Vec<(NWAStateID, Weight)> = Vec::new();
            let mut edge_accum_weight = Weight::zeros();

            while i < raw.len() && raw[i].0 == current_lbl {
                let target = raw[i].1;

                // Aggregate weights for the same target
                let mut target_w = raw[i].2.clone();
                i += 1;

                while i < raw.len() && raw[i].0 == current_lbl && raw[i].1 == target {
                    target_w |= &raw[i].2;
                    i += 1;
                }

                // Add to total edge weight
                edge_accum_weight |= &target_w;
                next_subset_raw.push((target, target_w));
            }

            // `next_subset_raw` is already sorted by NWAStateID because of `raw` sorting.
            // 4. Compute epsilon closure of this new subset.
            let next_closure = epsilon_closure_optimized(&self.nwa.states, &next_subset_raw);

            // 5. Register the new DWA state.
            let target_dwa_id = self.register_closure(next_closure);

            // 6. Add transition to DWA.
            let _ = self.dwa.add_transition(sid, current_lbl, target_dwa_id, edge_accum_weight);
        }
    }
}

fn try_build_singleton_loop_union(nwa: &NWA) -> Option<DWA> {
    if nwa.states.0.is_empty() || nwa.body.start_states.len() != 1 {
        return None;
    }

    let start = nwa.body.start_states[0];
    if start >= nwa.states.len() {
        return None;
    }

    if !nwa.states[start].transitions.is_empty() {
        return None;
    }

    let mut seed: WeightedSubset = Vec::new();
    seed.push((start, Weight::all()));
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

impl NWA {
    pub fn determinize_to_dwa2(&self) -> DWA {
        let custom_dwa = if let Some(dwa) = try_build_singleton_loop_union(self) {
            dwa
        } else {
            const STATE_LIMIT: usize = usize::MAX;
            crate::debug!(5, "Determinization: Using general-purpose subset construction (fast-path not taken).");

            if self.states.0.is_empty() {
                DWA::new()
            } else {
                let show_pbar = self.states.len() > 10000;
                let mp = if show_pbar { Some(MultiProgress::new()) } else { None };
                let main_pb = mp.as_ref().map(|mp_instance| {
                    let pb = mp_instance.add(ProgressBar::new(1));
                    pb.set_style(
                        ProgressStyle::default_bar()
                            .template(
                                "{spinner:.green} [{elapsed_precise}] States: \
                                 [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}",
                            )
                            .unwrap()
                            .progress_chars("#>-"),
                    );
                    pb.set_message("Determinizing NWA");
                    pb
                });

                let mut det = Determinizer::new(self, mp);
                let mut start_subset: WeightedSubset = Vec::new();
                for &s in &self.body.start_states {
                    start_subset.push((s, Weight::all()));
                }
                // Ensure input to closure is sorted (though start_states typically small)
                start_subset.sort_unstable_by(|a, b| a.0.cmp(&b.0));

                let start_closure = epsilon_closure_optimized(&self.states, &start_subset);
                let start_id = det.register_closure(start_closure);
                det.dwa.body.start_state = start_id;

                let mut processed_count = 0;
                while let Some(sid) = det.queue.pop_front() {
                    if det.seen.len() > STATE_LIMIT {
                        let timestamp = Local::now().format("%Y%m%d-%H%M%S");
                        let filename = format!("nwa_dump_{}.json", timestamp);
                        crate::debug!(
                            5,
                            "Determinization state limit ({}) exceeded. Dumping NWA to {} and panicking.",
                            STATE_LIMIT,
                            filename
                        );
                        let f = std::fs::File::create(&filename).expect("Unable to create dump file");
                        serde_json::to_writer(f, self).expect("Unable to write NWA to file");
                        panic!("Determinization aborted after reaching {} states.", STATE_LIMIT);
                    }

                    if let Some(pb) = &main_pb {
                        let total_states = det.seen.len();
                        pb.set_length(total_states as u64);
                        pb.set_position(processed_count as u64);
                        pb.set_message(format!("Expanding state {}/{}", processed_count + 1, total_states));
                    }

                    det.expand_state(sid);
                    processed_count += 1;
                }
                if let Some(pb) = main_pb {
                    pb.finish_with_message("Determinization complete");
                }
                det.dwa
            }
        };

        if DETERMINIZE_DEBUG {
            let rustfst_dwa = self.determinize_to_dwa_with_rustfst();
            crate::debug!(5, "[DETERMINIZE_DEBUG] Comparing custom determinization with rustfst...");
            crate::debug!(5, "[DETERMINIZE_DEBUG] Input NWA: {}", self);
            crate::debug!(5, "[DETERMINIZE_DEBUG] Custom DWA stats: {}", custom_dwa.stats());
            crate::debug!(5, "[DETERMINIZE_DEBUG] Rustfst DWA stats: {}", rustfst_dwa.stats());
            test_weighted_automata::stochastic_equivalence_test(custom_dwa.clone(), rustfst_dwa);
            crate::debug!(5, "[DETERMINIZE_DEBUG] Stochastic equivalence test passed.");
        }

        custom_dwa
    }

    pub fn determinize(&self) -> DWA {
        self.determinize_to_dwa2()
    }
}
