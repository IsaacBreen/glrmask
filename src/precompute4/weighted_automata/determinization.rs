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

/// Computes epsilon closure for a specific subset on the fly.
/// Used by the heuristic singleton check.
fn epsilon_closure_optimized(nwa_states: &NWAStates, seed: &WeightedSubset) -> WeightedSubset {
    let mut closure_map: HashMap<NWAStateID, Weight> = HashMap::with_capacity(seed.len() * 2);
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
        
        for (v, w_eps) in &nwa_states[u].epsilons {
            let cand = &uw & w_eps;
            if cand.is_empty() { continue; }

            let entry = closure_map.entry(*v).or_insert_with(Weight::zeros);
            
            // If the candidate adds new info (is not subset of existing), propagate
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

/// Precomputes the epsilon closure for every state in the NWA.
/// Returns a vector where index `i` contains the reachable states and weights from state `i` via epsilon transitions.
/// The result for each state includes the state itself with Weight::all() (identity).
fn precompute_all_epsilon_closures(states: &NWAStates) -> Vec<WeightedSubset> {
    let n = states.len();
    let mut reachability = Vec::with_capacity(n);

    for start_node in 0..n {
        let mut dists: HashMap<NWAStateID, Weight> = HashMap::new();
        let mut queue: VecDeque<NWAStateID> = VecDeque::new();

        // Self-reachability is identity
        dists.insert(start_node, Weight::all());
        queue.push_back(start_node);

        while let Some(u) = queue.pop_front() {
            let w_u = dists.get(&u).unwrap().clone();
            
            if u < n {
                for (v, w_eps) in &states[u].epsilons {
                    let new_w = &w_u & w_eps;
                    if new_w.is_empty() { continue; }

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

struct Determinizer<'a> {
    nwa: &'a NWA,
    eps_reach: &'a [WeightedSubset],
    
    // Map from canonical closure (Sorted Vec) to DWA State ID
    seen: HashMap<WeightedSubset, usize>,
    queue: VecDeque<usize>,
    // Store the closure for each DWA state
    closures: Vec<WeightedSubset>,
    
    dwa: DWA,
    mp: Option<MultiProgress>,
}

impl<'a> Determinizer<'a> {
    fn new(nwa: &'a NWA, eps_reach: &'a [WeightedSubset], mp: Option<MultiProgress>) -> Self {
        let mut dwa = DWA::new();
        dwa.states.0.clear();
        dwa.body.start_state = 0;
        Determinizer {
            nwa,
            eps_reach,
            seen: HashMap::new(),
            queue: VecDeque::new(),
            closures: Vec::new(),
            dwa,
            mp,
        }
    }

    fn register_closure(&mut self, closure: WeightedSubset) -> usize {
        if let Some(&id) = self.seen.get(&closure) {
            return id;
        }

        let id = self.dwa.add_state();

        // Compute final weight for this new DWA state
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

        // Transitions accumulation: Label -> TargetNWA -> Weight
        // Use BTreeMap for labels to keep them sorted (cleaner DWA), HashMap for targets for speed.
        let mut transitions: BTreeMap<Label, HashMap<NWAStateID, Weight>> = BTreeMap::new();
        let mut edge_weights: HashMap<Label, Weight> = HashMap::new();

        // 1. Collect outgoing labeled transitions from the subset.
        //    We perform intersection (w_u & w_trans) here.
        for (u, w_u) in &closure {
            let st = &self.nwa.states[*u];
            for (lbl, targets) in &st.transitions {
                if targets.is_empty() { continue; }

                let target_map = transitions.entry(*lbl).or_default();
                let edge_acc = edge_weights.entry(*lbl).or_insert_with(Weight::zeros);

                for (v, w_trans) in targets {
                    let w_out = w_u & w_trans;
                    if !w_out.is_empty() {
                        *edge_acc |= &w_out;
                        
                        target_map.entry(*v)
                            .and_modify(|w| *w |= &w_out)
                            .or_insert(w_out);
                    }
                }
            }
        }

        // 2. For each label, compute the epsilon-closed destination subset.
        //    We use the precomputed `eps_reach` to avoid traversing the NWA graph again.
        for (lbl, raw_targets) in transitions {
            let w_edge = edge_weights.remove(&lbl).unwrap();

            let mut dest_map: HashMap<NWAStateID, Weight> = HashMap::new();

            // Apply epsilon closure: 
            // Destination = Union_{ t in raw_targets } ( eps_reach[t] intersected with weight(t) )
            for (t, w_t) in raw_targets {
                if t < self.eps_reach.len() {
                    for (v_reach, w_reach) in &self.eps_reach[t] {
                        let combined = &w_t & w_reach;
                        if !combined.is_empty() {
                            dest_map.entry(*v_reach)
                                .and_modify(|w| *w |= &combined)
                                .or_insert(combined);
                        }
                    }
                }
            }

            let mut dest_subset: WeightedSubset = dest_map.into_iter().collect();
            dest_subset.sort_unstable_by(|a, b| a.0.cmp(&b.0));

            let dest_dwa_id = self.register_closure(dest_subset);
            let _ = self.dwa.add_transition(sid, lbl, dest_dwa_id, w_edge);
        }
    }
}

fn try_build_singleton_loop_union(nwa: &NWA) -> Option<DWA> {
    if nwa.states.0.is_empty() || nwa.body.start_states.len() != 1 {
        return None;
    }

    let start = nwa.body.start_states[0];
    if start >= nwa.states.len() { return None; }

    if !nwa.states[start].transitions.is_empty() {
        return None;
    }

    let mut seed: WeightedSubset = Vec::new();
    seed.push((start, Weight::all()));
    // Use the local helper here to avoid precomputing everything for this fast path
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
        if let Some(dwa) = try_build_singleton_loop_union(self) {
            return dwa;
        }

        const STATE_LIMIT: usize = 250_000; 
        
        if self.states.0.is_empty() {
            return DWA::new();
        }

        crate::debug!(5, "Determinization: Precomputing epsilon closures...");
        let eps_reach = precompute_all_epsilon_closures(&self.states);

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

        let mut det = Determinizer::new(self, &eps_reach, mp);

        // Construct start subset by applying epsilon closure to start states
        let mut start_map: HashMap<NWAStateID, Weight> = HashMap::new();
        for &s in &self.body.start_states {
            if s < eps_reach.len() {
                // Start states have weight ALL initially, so we just take the epsilon closure directly.
                // eps_reach[s] contains pairs (v, path_weight). 
                // Since initial weight is ALL, combined weight is just path_weight.
                for (v, w_reach) in &eps_reach[s] {
                    start_map.entry(*v)
                        .and_modify(|acc| *acc |= w_reach)
                        .or_insert_with(|| w_reach.clone());
                }
            }
        }
        
        let mut start_subset: WeightedSubset = start_map.into_iter().collect();
        start_subset.sort_unstable_by(|a, b| a.0.cmp(&b.0));

        let start_id = det.register_closure(start_subset);
        det.dwa.body.start_state = start_id;

        let mut processed_count = 0;
        while let Some(sid) = det.queue.pop_front() {
            if det.seen.len() > STATE_LIMIT {
                let timestamp = Local::now().format("%Y%m%d-%H%M%S");
                let filename = format!("nwa_dump_{}.json", timestamp);
                crate::debug!(5, "Determinization state limit ({}) exceeded. Dumping NWA to {} and panicking.", STATE_LIMIT, filename);
                let f = std::fs::File::create(&filename).expect("Unable to create dump file");
                serde_json::to_writer(f, self).expect("Unable to write NWA to file");
                panic!("Determinization aborted after reaching {} states.", STATE_LIMIT);
            }

            if let Some(pb) = &main_pb {
                if processed_count % 100 == 0 {
                    let total_states = det.seen.len();
                    pb.set_length(total_states as u64);
                    pb.set_position(processed_count as u64);
                    pb.set_message(format!("Expanding state {}/{}", processed_count + 1, total_states));
                }
            }

            det.expand_state(sid);
            processed_count += 1;
        }
        if let Some(pb) = main_pb {
            pb.finish_with_message("Determinization complete");
        }

        if DETERMINIZE_DEBUG {
            let rustfst_dwa = self.determinize_to_dwa_with_rustfst();
            crate::debug!(5, "[DETERMINIZE_DEBUG] Comparing custom determinization with rustfst...");
            test_weighted_automata::stochastic_equivalence_test(det.dwa.clone(), rustfst_dwa);
        }

        det.dwa
    }

    pub fn determinize(&self) -> DWA {
        self.determinize_to_dwa2()
    }
}
