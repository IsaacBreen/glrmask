// src/precompute4/weighted_automata/determinization.rs

#![allow(dead_code)]
#![allow(clippy::needless_borrow)]

use chrono::Local;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};

use crate::precompute4::test_weighted_automata;
use super::common::{DETERMINIZE_DEBUG, NWAStateID, Weight};
use super::dwa::DWA;
use super::nwa::{NWA, NWAStates};

type WeightedSubset = BTreeMap<NWAStateID, Weight>;
type ClosureMap = BTreeMap<NWAStateID, Weight>;

fn weight_union(a: Weight, b: &Weight) -> Weight { a | b.clone() }
fn weight_union_in_place(dst: &mut Weight, src: &Weight) { *dst |= src; }
fn weight_intersection(a: &Weight, b: &Weight) -> Weight { a & b }
fn is_zero(w: &Weight) -> bool { w.is_empty() }

fn epsilon_closure(nwa_states: &NWAStates, seed: &WeightedSubset) -> ClosureMap {
    let mut closure: ClosureMap = ClosureMap::new();
    let mut queue: VecDeque<NWAStateID> = VecDeque::new();

    for (sid, w) in seed {
        if !is_zero(w) {
            let prev = closure.get(sid).cloned().unwrap_or_else(Weight::zeros);
            let neww = weight_union(prev.clone(), w);
            if neww != prev {
                closure.insert(*sid, neww.clone());
                queue.push_back(*sid);
            }
        }
    }

    while let Some(u) = queue.pop_front() {
        let uw = closure.get(&u).cloned().unwrap_or_else(Weight::zeros);
        if is_zero(&uw) {
            continue;
        }
        for (v, w_eps) in &nwa_states[u].epsilons {
            let cand = weight_intersection(&uw, w_eps);
            if is_zero(&cand) {
                continue;
            }
            let prev = closure.get(v).cloned().unwrap_or_else(Weight::zeros);
            let merged = weight_union(prev.clone(), &cand);
            if merged != prev {
                closure.insert(*v, merged.clone());
                queue.push_back(*v);
            }
        }
    }

    closure
}

fn collect_labels(nwa_states: &NWAStates, closure: &ClosureMap) -> BTreeSet<i16> {
    let mut labels: BTreeSet<i16> = BTreeSet::new();
    for (sid, cw) in closure {
        if is_zero(cw) {
            continue;
        }
        labels.extend(nwa_states[*sid].transitions.keys());
    }
    labels
}

fn next_subset_for_label(nwa_states: &NWAStates, closure: &ClosureMap, ch: i16) -> WeightedSubset {
    let mut next: WeightedSubset = WeightedSubset::new();
    for (sid, cw) in closure {
        if is_zero(cw) {
            continue;
        }
        let st = &nwa_states[*sid];
        if let Some(targets) = st.transitions.get(&ch) {
            for (to, w_edge) in targets {
                let cand = weight_intersection(cw, w_edge);
                if is_zero(&cand) {
                    continue;
                }
                next.entry(*to)
                    .and_modify(|w| weight_union_in_place(w, &cand))
                    .or_insert(cand);
            }
        }
    }
    next.retain(|_, w| !is_zero(w));
    next
}

fn union_over_values(map: &WeightedSubset) -> Weight {
    let mut acc = Weight::zeros();
    for w in map.values() {
        acc = acc | w.clone();
    }
    acc
}

fn compute_state_and_final_weights(nwa: &NWA, closure: &ClosureMap) -> (Option<Weight>, Option<Weight>) {
    let entry_opt = None;
    let mut finalw = Weight::zeros();
    for (sid, cw) in closure {
        if let Some(fw) = &nwa.states[*sid].final_weight {
            let cand = cw & fw;
            if !is_zero(&cand) {
                finalw = finalw | cand;
            }
        }
    }
    let final_opt = if finalw.is_empty() { None } else { Some(finalw) };
    (entry_opt, final_opt)
}

struct Determinizer<'a> {
    nwa: &'a NWA,
    seen: HashMap<ClosureMap, usize>,
    subsets: Vec<WeightedSubset>,
    closures: Vec<ClosureMap>,
    queue: VecDeque<usize>,
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
            subsets: Vec::new(),
            closures: Vec::new(),
            queue: VecDeque::new(),
            dwa,
            mp,
        }
    }

    fn register_state(&mut self, subset: WeightedSubset) -> usize {
        let mut subset_clean: WeightedSubset = WeightedSubset::new();
        for (sid, w) in subset {
            if !is_zero(&w) {
                subset_clean.insert(sid, w);
            }
        }

        let closure = epsilon_closure(&self.nwa.states, &subset_clean);
        if let Some(&id) = self.seen.get(&closure) {
            return id;
        }

        let id = self.dwa.add_state();
        let (_entry_opt, final_opt) = compute_state_and_final_weights(self.nwa, &closure);
        if let Some(w) = final_opt {
            let _ = self.dwa.set_final_weight(id, w);
        }

        self.seen.insert(closure.clone(), id);
        self.subsets.push(subset_clean);
        self.closures.push(closure);
        self.queue.push_back(id);
        id
    }

    fn expand_state(&mut self, sid: usize) {
        let closure = &self.closures[sid];
        if closure.is_empty() {
            return;
        }

        let labels_pb = if let Some(mp) = &self.mp {
            let pb = mp.add(ProgressBar::new(0));
            pb.set_style(
                ProgressStyle::default_bar()
                    .template(
                        "  {spinner:.green} [{elapsed_precise}] Labels: \
                         [{bar:40.cyan/blue}] {pos}/{len} {msg}",
                    )
                    .unwrap()
                    .progress_chars("#>-"),
            );
            pb.set_message(format!("State {}", sid));
            Some(pb)
        } else {
            None
        };

        let labels = collect_labels(&self.nwa.states, closure);
        if let Some(pb) = &labels_pb {
            pb.set_length(labels.len() as u64);
        }

        let mut transition_data: BTreeMap<i16, (WeightedSubset, Weight)> = BTreeMap::new();
        for ch in &labels {
            let sub_ch = next_subset_for_label(&self.nwa.states, closure, *ch);
            let w_ch = union_over_values(&sub_ch);
            if !sub_ch.is_empty() && !is_zero(&w_ch) {
                transition_data.insert(*ch, (sub_ch, w_ch));
            }
            if let Some(pb) = &labels_pb {
                pb.inc(1);
            }
        }

        if let Some(pb) = &labels_pb {
            pb.set_message(format!("State {}: installing transitions", sid));
            pb.set_length(transition_data.len() as u64);
            pb.set_position(0);
        }

        for (ch, (sub_ch, w_ch)) in transition_data {
            let to_ch_id = self.register_state(sub_ch);
            let _ = self.dwa.add_transition(sid, ch, to_ch_id, w_ch.clone()).ok();
            if let Some(pb) = &labels_pb {
                pb.inc(1);
            }
        }

        if let Some(pb) = labels_pb {
            pb.finish_and_clear();
        }
    }
}

fn try_build_singleton_loop_union(nwa: &NWA) -> Option<DWA> {
    if nwa.states.0.is_empty() || nwa.body.start_state >= nwa.states.len() {
        return None;
    }

    let start = nwa.body.start_state;
    if !nwa.states[start].transitions.is_empty() {
        return None;
    }

    let mut seed: WeightedSubset = WeightedSubset::new();
    seed.insert(start, Weight::all());
    let start_closure = epsilon_closure(&nwa.states, &seed);

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

    let mut label_to_weight: BTreeMap<i16, Weight> = BTreeMap::new();
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
            crate::debug!(
                5,
                "[DEBUG] Determinization: Using general-purpose subset construction (fast-path not taken)."
            );

            if self.states.0.is_empty() || self.body.start_state >= self.states.len() {
                DWA::new()
            } else {
                let mp = MultiProgress::new();
                let main_pb = mp.add(ProgressBar::new(1));
                main_pb.set_style(
                    ProgressStyle::default_bar()
                        .template(
                            "{spinner:.green} [{elapsed_precise}] States: \
                             [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}",
                        )
                        .unwrap()
                        .progress_chars("#>-"),
                );
                main_pb.set_message("Determinizing NWA");

                let mut det = Determinizer::new(self, Some(mp));
                let mut start_subset: WeightedSubset = WeightedSubset::new();
                start_subset.insert(self.body.start_state, Weight::all());
                let start_id = det.register_state(start_subset);
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

                    let total_states = det.seen.len();
                    main_pb.set_length(total_states as u64);
                    main_pb.set_position(processed_count as u64);
                    main_pb.set_message(format!("Expanding state {}/{}", processed_count + 1, total_states));

                    det.expand_state(sid);
                    processed_count += 1;
                }
                main_pb.finish_with_message("Determinization complete");
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
}
