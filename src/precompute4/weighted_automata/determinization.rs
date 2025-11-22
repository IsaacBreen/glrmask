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

// Invariants: strictly sorted by NWAStateID.
type WeightedSubset = Vec<(NWAStateID, Weight)>;

/// Represents a pre-calculated transition u --(Label)--> v
/// where the epsilon closure has already been applied to v.
#[derive(Clone, Debug)]
struct LutEntry {
    label: Label,
    target: NWAStateID,
    weight: Weight,
}

struct FastDeterminizer {
    // Maps NWA State ID -> List of outgoing epsilon-closed transitions
    // Sorted by (label, target) for fast merging
    lut: Vec<Vec<LutEntry>>,

    // Maps NWA State ID -> Combined final weight reachable via epsilon
    effective_final_weights: Vec<Option<Weight>>,

    // Determinization state
    seen: HashMap<WeightedSubset, usize>,
    queue: VecDeque<usize>,
    closures: Vec<WeightedSubset>, // Map DWA ID -> Subset

    dwa: DWA,

    // Reusable buffers
    batch_buffer: Vec<LutEntry>,
}

impl FastDeterminizer {
    fn new(nwa: &NWA, mp: Option<MultiProgress>) -> Self {
        // 1. Precompute Epsilon Closures (BFS/Dijkstra for each state)
        //    Result: reach[u] = list of (v, path_weight)
        let n = nwa.states.len();
        let mut eps_reach = vec![Vec::new(); n];

        // Using a temporary buffer for BFS to avoid repeated allocations
        let mut dists = vec![None; n];
        let mut dirty = Vec::with_capacity(n);
        let mut q = VecDeque::with_capacity(64);

        for start_node in 0..n {
            // Self-reachability
            dists[start_node] = Some(Weight::all());
            dirty.push(start_node);
            q.push_back(start_node);

            while let Some(u) = q.pop_front() {
                let w_u = dists[u].clone().unwrap(); // Safe, u is in dirty

                // Propagate
                for (v, w_eps) in &nwa.states[u].epsilons {
                    if *v >= n { continue; }
                    let w_new = &w_u & w_eps;
                    if w_new.is_empty() { continue; }

                    match &mut dists[*v] {
                        Some(w_v) => {
                            if !w_new.is_subset_of(w_v) {
                                *w_v |= &w_new;
                                q.push_back(*v);
                            }
                        }
                        None => {
                            dists[*v] = Some(w_new);
                            dirty.push(*v);
                            q.push_back(*v);
                        }
                    }
                }
            }

            // Save results
            let mut reach_vec = Vec::with_capacity(dirty.len());
            for &node in &dirty {
                if let Some(w) = dists[node].take() {
                    reach_vec.push((node, w));
                }
            }
            dirty.clear(); // dists is already cleared by take()
            // Sort by ID for canonical consistency
            reach_vec.sort_unstable_by(|a, b| a.0.cmp(&b.0));
            eps_reach[start_node] = reach_vec;
        }

        // 2. Precompute Effective Final Weights
        //    final[u] = OR_{v in reach[u]} (reach_weight(u,v) & nwa.final[v])
        let mut effective_final_weights = Vec::with_capacity(n);
        for u in 0..n {
            let mut acc = Weight::zeros();
            for (v, w_path) in &eps_reach[u] {
                if let Some(fw) = &nwa.states[*v].final_weight {
                    acc |= &(w_path & fw);
                }
            }
            effective_final_weights.push(if acc.is_empty() { None } else { Some(acc) });
        }

        // 3. Precompute Look-Up Table (LUT)
        //    For every transition u --L--> v in NWA,
        //    combine with v --eps--> z in eps_reach[v].
        //    Result: u --L--> z with weight (w_trans & w_eps_path)
        let mut lut = vec![Vec::new(); n];

        // Progress bar for LUT generation if needed (usually fast enough, but good for debug)
        let pb = mp.as_ref().map(|m| {
            let p = m.add(ProgressBar::new(n as u64));
            p.set_message("Precomputing LUT");
            p
        });

        for u in 0..n {
            let st = &nwa.states[u];
            let mut u_transitions = Vec::new();

            for (&label, targets) in &st.transitions {
                for (v, w_trans) in targets {
                    if *v >= n { continue; }
                    // Expand v via epsilon closure
                    for (z, w_eps) in &eps_reach[*v] {
                        let w_combined = w_trans & w_eps;
                        if !w_combined.is_empty() {
                            u_transitions.push(LutEntry {
                                label,
                                target: *z,
                                weight: w_combined,
                            });
                        }
                    }
                }
            }

            // Sort LUT for deterministic processing and easier merging
            // Sort by Label, then Target
            u_transitions.sort_unstable_by(|a, b| {
                match a.label.cmp(&b.label) {
                    Ordering::Equal => a.target.cmp(&b.target),
                    other => other,
                }
            });

            // Optional: Compact the LUT by merging duplicates (u --L--> z appearing multiple times)
            // This happens if u has multiple transitions on L to {v1, v2} and both reach z.
            let mut compacted = Vec::with_capacity(u_transitions.len());
            if !u_transitions.is_empty() {
                let mut iter = u_transitions.into_iter();
                let mut cur = iter.next().unwrap();

                for next in iter {
                    if next.label == cur.label && next.target == cur.target {
                        cur.weight |= &next.weight;
                    } else {
                        if !cur.weight.is_empty() {
                            compacted.push(cur);
                        }
                        cur = next;
                    }
                }
                if !cur.weight.is_empty() {
                    compacted.push(cur);
                }
            }

            lut[u] = compacted;

            if let Some(ref p) = pb {
                if u % 1000 == 0 { p.inc(1000); }
            }
        }
        if let Some(p) = pb { p.finish(); }

        let mut dwa = DWA::new();
        dwa.states.0.clear();

        FastDeterminizer {
            lut,
            effective_final_weights,
            seen: HashMap::new(),
            queue: VecDeque::new(),
            closures: Vec::new(),
            dwa,
            batch_buffer: Vec::with_capacity(4096),
        }
    }

    fn register_subset(&mut self, subset: WeightedSubset) -> usize {
        if let Some(&id) = self.seen.get(&subset) {
            return id;
        }

        let id = self.dwa.add_state();

        // Calculate DWA final weight using precomputed effective weights
        // DWA state final = OR_{u in subset} (w_u & effective_final[u])
        let mut final_acc = Weight::zeros();
        for (u, w_u) in &subset {
            if let Some(fw) = &self.effective_final_weights[*u] {
                final_acc |= &(w_u & fw);
            }
        }
        if !final_acc.is_empty() {
            let _ = self.dwa.set_final_weight(id, final_acc);
        }

        self.seen.insert(subset.clone(), id);
        self.closures.push(subset);
        self.queue.push_back(id);
        id
    }

    fn expand(&mut self, id: usize) {
        // We take the subset out by index. Cloning the vector is necessary
        // but these subsets are usually small (sparse).
        let subset = self.closures[id].clone();

        self.batch_buffer.clear();

        // 1. Gather all potential transitions
        //    Since `lut[u]` contains fully expanded u --L--> z transitions,
        //    we just need to apply the subset weight `w_u` to them.
        for (u, w_u) in &subset {
            if *u >= self.lut.len() { continue; }

            // optimization: if w_u is empty, skip (shouldn't happen in valid subset)

            for entry in &self.lut[*u] {
                let w_edge = w_u & &entry.weight;
                if !w_edge.is_empty() {
                    self.batch_buffer.push(LutEntry {
                        label: entry.label,
                        target: entry.target,
                        weight: w_edge,
                    });
                }
            }
        }

        if self.batch_buffer.is_empty() {
            return;
        }

        // 2. Sort gathered transitions by (Label, Target) to group them
        self.batch_buffer.sort_unstable_by(|a, b| {
            match a.label.cmp(&b.label) {
                Ordering::Equal => a.target.cmp(&b.target),
                other => other,
            }
        });

        // 3. Aggregate and emit
        //    Iterate through batch. All entries with same Label form a transition edge.
        //    Inside that Label group, all entries with same Target form a component of the dest subset.
        let len = self.batch_buffer.len();
        let mut i = 0;
        while i < len {
            let current_label = self.batch_buffer[i].label;
            let mut j = i;

            let mut edge_weight_acc = Weight::zeros();
            let mut next_subset = Vec::new();

            // Process all entries for `current_label`
            while j < len && self.batch_buffer[j].label == current_label {
                let current_target = self.batch_buffer[j].target;
                let mut k = j;

                // Accumulate weights for specific target z
                let mut target_weight_acc = Weight::zeros();
                while k < len && self.batch_buffer[k].label == current_label && self.batch_buffer[k].target == current_target {
                    let w = &self.batch_buffer[k].weight;
                    target_weight_acc |= w;
                    // Also accumulate to the total edge weight for this label
                    // Note: DWA edge weight is Union(All contributions).
                    // Wait. In a valid DWA, the edge weight W_edge should satisfy:
                    // W_edge >= W_dest_component for all components.
                    // Actually, usually W_edge = Union(all dest components).
                    // We can just accumulate it here.
                    edge_weight_acc |= w;
                    k += 1;
                }

                if !target_weight_acc.is_empty() {
                    next_subset.push((current_target, target_weight_acc));
                }
                j = k;
            }

            // Register the new state
            // next_subset is already sorted by target because we sorted batch by target!
            // (We iterated distinct targets in increasing order)
            if !next_subset.is_empty() {
                let dest_id = self.register_subset(next_subset);
                let _ = self.dwa.add_transition(id, current_label, dest_id, edge_weight_acc);
            }

            i = j;
        }
    }
}

impl NWA {
    pub fn determinize_to_dwa2(&self) -> DWA {
        // Heuristic optimization for simple loops
        if let Some(dwa) = try_build_singleton_loop_union(self) {
            return dwa;
        }

        const STATE_LIMIT: usize = 250_000;

        crate::debug!(5, "Determinization (Fast): Starting...");
        let show_pbar = self.states.len() > 10000;
        let mp = if show_pbar { Some(MultiProgress::new()) } else { None };

        let mut det = FastDeterminizer::new(self, mp.clone());

        // Initial state = Closure(StartStates)
        // We can use the precomputed eps_reach here too.
        // Start states have weight ALL.
        // So StartSubset = Union_{s in start_states} eps_reach[s]

        // To merge efficiently, we collect all reach vectors and merge-sort or map-reduce.
        // Since N start states is small, a map is fine.
        let mut start_map = HashMap::new();
        for &s in &self.body.start_states {
            // Use the internal precomputed reach if we could access it,
            // but `det` owns it inside `lut` logic implicitly.
            // Wait, `det` discarded `eps_reach`. It only kept `lut` and `final`.
            // We need to re-run epsilon closure just for the start state?
            // That's unfortunate.
            // Fix: Let's just do a quick BFS for start states. It's one-time.
            let mut q = VecDeque::new();
            q.push_back(s);
            start_map.insert(s, Weight::all());
        }

        // Full BFS for start subset
        let mut q_bfs = VecDeque::new();
        for (&s, _) in &start_map { q_bfs.push_back(s); }

        while let Some(u) = q_bfs.pop_front() {
            let w_u = start_map[&u].clone();
            if u >= self.states.len() { continue; }
            for (v, w_eps) in &self.states[u].epsilons {
                let w_new = &w_u & w_eps;
                if !w_new.is_empty() {
                    let entry = start_map.entry(*v).or_insert_with(Weight::zeros);
                    if !w_new.is_subset_of(entry) {
                        *entry |= &w_new;
                        q_bfs.push_back(*v);
                    }
                }
            }
        }

        let mut start_subset: WeightedSubset = start_map.into_iter().collect();
        start_subset.sort_unstable_by(|a, b| a.0.cmp(&b.0));

        let start_id = det.register_subset(start_subset);
        det.dwa.body.start_state = start_id;

        // Main Loop
        let main_pb = mp.as_ref().map(|m| {
            let p = m.add(ProgressBar::new(STATE_LIMIT as u64));
            p.set_style(ProgressStyle::default_bar().template("{spinner:.green} [{elapsed_precise}] DWA States: {pos} ({msg})").unwrap());
            p
        });

        let mut count = 0;
        while let Some(id) = det.queue.pop_front() {
            if det.seen.len() > STATE_LIMIT {
                 panic!("Determinization state limit exceeded");
            }
            if count % 1000 == 0 {
                 if let Some(ref p) = main_pb {
                     p.set_position(det.seen.len() as u64);
                 }
            }
            det.expand(id);
            count += 1;
        }

        if let Some(p) = main_pb { p.finish_with_message("Done"); }

        det.dwa
    }

    pub fn determinize(&self) -> DWA {
        self.determinize_to_dwa2()
    }

    pub fn _determinize(&self) -> DWA {
        self.determinize_to_dwa2()
    }
}

// Re-include the heuristic helper to ensure compilation
fn try_build_singleton_loop_union(nwa: &NWA) -> Option<DWA> {
    if nwa.states.0.is_empty() || nwa.body.start_states.len() != 1 {
        return None;
    }
    let start = nwa.body.start_states[0];
    if start >= nwa.states.len() { return None; }
    if !nwa.states[start].transitions.is_empty() { return None; }

    let mut start_closure = Vec::new();
    let mut q = VecDeque::new();
    let mut visited = HashMap::new();
    visited.insert(start, Weight::all());
    q.push_back(start);

    while let Some(u) = q.pop_front() {
        let w_u = visited[&u].clone();
        if u < nwa.states.len() {
            for (v, w_eps) in &nwa.states[u].epsilons {
                let w_new = &w_u & w_eps;
                if !w_new.is_empty() {
                    let entry = visited.entry(*v).or_insert_with(Weight::zeros);
                    if !w_new.is_subset_of(entry) {
                        *entry |= &w_new;
                        q.push_back(*v);
                    }
                }
            }
        }
    }
    for (u, w) in visited { start_closure.push((u, w)); }

    let mut comps = Vec::new();
    for (sid, cw) in start_closure.iter() {
        if *sid == start || cw.is_empty() { continue; }
        let st = &nwa.states[*sid];
        if !st.epsilons.is_empty() { return None; }
        for (_lbl, vec_targets) in st.transitions.iter() {
            for (to, _) in vec_targets {
                if *to != *sid { return None; }
            }
        }
        if let Some(fw) = &st.final_weight {
            let base = cw & fw;
            if !base.is_empty() { comps.push((*sid, base)); }
        }
    }
    if comps.is_empty() { return None; }
    for i in 0..comps.len() {
        for j in (i + 1)..comps.len() {
            if !(comps[i].1.clone() & comps[j].1.clone()).is_empty() { return None; }
        }
    }

    let mut label_to_weight = BTreeMap::new();
    for (sid, base) in &comps {
        let st = &nwa.states[*sid];
        for (lbl, vec_targets) in st.transitions.iter() {
            let mut w_union = Weight::zeros();
            for (_to, w) in vec_targets { w_union |= w; }
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
    for (_sid, base) in &comps { final_union |= base; }

    let mut dwa = DWA::new();
    let s0 = dwa.body.start_state;
    if !final_union.is_empty() { let _ = dwa.set_final_weight(s0, final_union); }
    for (lbl, w) in label_to_weight {
        if !w.is_empty() { let _ = dwa.add_transition(s0, lbl, s0, w); }
    }
    Some(dwa)
}