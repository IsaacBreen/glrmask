#![allow(clippy::needless_borrow)]
#![allow(clippy::type_complexity)]


use std::collections::{BTreeMap, HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use crate::precompute4::weighted_automata::{DWA, DWAState, DWAStates, DWABody, StateID, Weight};
use crate::precompute4::weighted_automata::common::Label;

impl DWA {
    pub fn minimize_acyclic(&mut self) {
        let opts = ExactMinimizeOpts::default();
        if let Ok(minimized) = minimize_acyclic_dwa_exact(self, opts) {
            *self = minimized;
        } else {
            panic!("minimize_acyclic_dwa_exact failed");
        }
    }
}

#[derive(Debug, Clone)]
pub enum DWAMinimizeError {
    NotAcyclic,
    StartOutOfBounds { start: StateID, n: usize },
}

/// Options for the exact minimizer.
///
/// If you want a guaranteed-minimal result, leave `node_limit` and `step_limit` as `None`.
#[derive(Debug, Clone)]
pub struct ExactMinimizeOpts {
    /// If set, abort search once we ever see more than this many states
    /// in the (trimmed) base automaton. (Safety valve.)
    pub node_limit: Option<usize>,
    /// If set, abort search after this many DFS nodes. (Safety valve.)
    pub step_limit: Option<u64>,
}

impl Default for ExactMinimizeOpts {
    fn default() -> Self {
        Self {
            node_limit: None,
            step_limit: None,
        }
    }
}

/// Public entry point: exact, provably minimal (merge-only) minimization for *acyclic* DWAs.
///
/// This returns the smallest equivalent quotient automaton (states merged, unreachable dropped),
/// under `DWA::eval_word_weight` semantics.
///
/// Worst-case exponential (NP-hard), but correct and optimal when it terminates.
///
/// Pipeline:
/// 1) semantics-preserving dead-token trimming
/// 2) strict DAG minimization (fast)
/// 3) exact search for additional merges (slow, optimal)
pub fn minimize_acyclic_dwa_exact(input: &DWA, opts: ExactMinimizeOpts) -> Result<DWA, DWAMinimizeError> {
    if input.states.len() == 0 {
        return Ok(DWA::default());
    }
    if input.body.start_state >= input.states.len() {
        return Err(DWAMinimizeError::StartOutOfBounds { start: input.body.start_state, n: input.states.len() });
    }

    // 1) Dead-token trim (semantics-preserving).
    let trimmed = trim_dead_tokens_and_edges(input)?;

    // 2) Strict DAG minimization (polynomial; merges truly identical residual states).
    let base = minimize_strict_acyclic(&trimmed)?;

    if let Some(limit) = opts.node_limit {
        if base.states.len() > limit {
            // Still return something correct; but no exact search.
            return Ok(base);
        }
    }

    // 3) Exact search for additional merges.
    let exact = minimize_exact_by_merging(&base, opts.step_limit)?;
    Ok(exact)
}

/* ========================================================================================== */
/* == Step 1: semantics-preserving dead-token trimming (forward/backward) ==================== */
/* ========================================================================================== */

fn topo_order(dwa: &DWA) -> Result<Vec<StateID>, DWAMinimizeError> {
    let n = dwa.states.len();
    if n == 0 {
        return Ok(vec![]);
    }

    let mut indeg = vec![0usize; n];
    for u in 0..n {
        for &v in dwa.states[u].transitions.values() {
            if v < n {
                indeg[v] += 1;
            }
        }
    }

    let mut q = VecDeque::new();
    for i in 0..n {
        if indeg[i] == 0 {
            q.push_back(i);
        }
    }

    let mut order = Vec::with_capacity(n);
    while let Some(u) = q.pop_front() {
        order.push(u);
        for &v in dwa.states[u].transitions.values() {
            if v >= n {
                continue;
            }
            indeg[v] -= 1;
            if indeg[v] == 0 {
                q.push_back(v);
            }
        }
    }

    if order.len() != n {
        return Err(DWAMinimizeError::NotAcyclic);
    }
    Ok(order)
}

/// Trim unreachable states (graph reachability). This is safe for your semantics because
/// unreachable states can never be visited from the start on any word.
fn trim_unreachable(mut dwa: DWA) -> Result<DWA, DWAMinimizeError> {
    let n = dwa.states.len();
    if n == 0 {
        return Ok(dwa);
    }
    let start = dwa.body.start_state;
    if start >= n {
        return Err(DWAMinimizeError::StartOutOfBounds { start, n });
    }

    let mut seen = vec![false; n];
    let mut stack = vec![start];
    seen[start] = true;

    while let Some(u) = stack.pop() {
        for &v in dwa.states[u].transitions.values() {
            if v >= n {
                continue;
            }
            if !seen[v] {
                seen[v] = true;
                stack.push(v);
            }
        }
    }

    // Remap old -> new ids
    let mut map = vec![None; n];
    let mut new_states = Vec::new();
    for i in 0..n {
        if seen[i] {
            map[i] = Some(new_states.len());
            new_states.push(dwa.states[i].clone());
        }
    }
    let new_start = map[start].unwrap();

    // Remap transitions
    for st in &mut new_states {
        let labels: Vec<Label> = st.transitions.keys().copied().collect();
        for lbl in labels {
            let to = st.transitions[&lbl];
            match map.get(to).and_then(|x| *x) {
                Some(new_to) => {
                    st.transitions.insert(lbl, new_to);
                }
                None => {
                    st.transitions.remove(&lbl);
                    st.trans_weights.remove(&lbl);
                }
            }
        }

        // Drop weights that have no target transition (these don't matter for eval_word_weight).
        let default_labels: Vec<Label> = st.trans_weights
            .keys()
            .filter(|l| !st.transitions.contains_key(l))
            .copied()
            .collect();
        for lbl in default_labels {
            st.trans_weights.remove(&lbl);
        }
    }

    dwa.states = DWAStates(new_states);
    dwa.body.start_state = new_start;
    Ok(dwa)
}

/// Semantics-preserving trimming:
///
/// Let `forward[u]` = tokens that can reach `u` from start while surviving intersections.
/// Let `backward[u]` = tokens that can be accepted from `u` by some suffix.
///
/// Then any token outside these “live” sets is irrelevant to *all* outputs from the start.
/// We can safely replace each transition weight by:
///     w' = forward[u] ∩ w(u,a) ∩ backward[v]
/// and each final weight by:
///     fw' = forward[u] ∩ fw(u)
///
/// and drop empty edges/finals.
fn trim_dead_tokens_and_edges(input: &DWA) -> Result<DWA, DWAMinimizeError> {
    let n = input.states.len();
    if n == 0 {
        return Ok(input.clone());
    }
    let start = input.body.start_state;
    if start >= n {
        return Err(DWAMinimizeError::StartOutOfBounds { start, n });
    }

    let order = topo_order(input)?;
    let mut forward = vec![Weight::zeros(); n];
    forward[start] = Weight::all();

    for &u in &order {
        if forward[u].is_empty() {
            continue;
        }
        let fu = forward[u].clone();
        for (lbl, &v) in &input.states[u].transitions {
            if v >= n {
                continue;
            }
            let w = input.states[u]
                .trans_weights
                .get(lbl)
                .cloned()
                .unwrap_or_else(Weight::all);

            let mut flow = fu.clone();
            flow &= &w;
            if !flow.is_subset_of(&forward[v]) {
                forward[v] |= &flow;
            }
        }
    }

    let mut backward = vec![Weight::zeros(); n];
    for u in 0..n {
        if let Some(fw) = &input.states[u].final_weight {
            backward[u] |= fw;
        }
    }

    for &u in order.iter().rev() {
        let mut bu = backward[u].clone();
        for (lbl, &v) in &input.states[u].transitions {
            if v >= n {
                continue;
            }
            let w = input.states[u]
                .trans_weights
                .get(lbl)
                .cloned()
                .unwrap_or_else(Weight::all);

            let contrib = &w & &backward[v];
            if !contrib.is_subset_of(&bu) {
                bu |= &contrib;
            }
        }
        backward[u] = bu;
    }

    // Build trimmed automaton (same indexing, then trim unreachable).
    let mut new_states = vec![DWAState::default(); n];
    for u in 0..n {
        let old = &input.states[u];

        // Final
        if let Some(fw) = &old.final_weight {
            let mut nf = fw.clone();
            nf &= &forward[u];
            if !nf.is_empty() {
                new_states[u].final_weight = Some(nf);
            }
        }

        // Transitions
        for (lbl, &v) in &old.transitions {
            if v >= n {
                continue;
            }
            let w = old.trans_weights.get(lbl).cloned().unwrap_or_else(Weight::all);

            let mut nw = w;
            nw &= &forward[u];
            nw &= &backward[v];

            if nw.is_empty() {
                continue;
            }
            new_states[u].transitions.insert(*lbl, v);
            new_states[u].trans_weights.insert(*lbl, nw);
        }
    }

    let trimmed = DWA {
        states: DWAStates(new_states),
        body: DWABody { start_state: start },
    };

    trim_unreachable(trimmed)
}

/* ========================================================================================== */
/* == Step 2: strict acyclic minimization (bottom-up hashing) ================================ */
/* ========================================================================================== */

#[derive(Clone, Debug, Eq)]
enum WeightKey {
    All,
    Empty,
    Ranges(Vec<(usize, usize)>),
}

impl PartialEq for WeightKey {
    fn eq(&self, other: &Self) -> bool {
        use WeightKey::*;
        match (self, other) {
            (All, All) => true,
            (Empty, Empty) => true,
            (Ranges(a), Ranges(b)) => a == b,
            _ => false,
        }
    }
}

impl Hash for WeightKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            WeightKey::All => {
                0u8.hash(state);
            }
            WeightKey::Empty => {
                1u8.hash(state);
            }
            WeightKey::Ranges(rs) => {
                2u8.hash(state);
                rs.hash(state);
            }
        }
    }
}

fn weight_key(w: &Weight) -> WeightKey {
    if w.is_all_fast() {
        return WeightKey::All;
    }
    if w.is_empty() {
        return WeightKey::Empty;
    }
    // This relies on Weight exposing `rsb.ranges()` like your JSON exporter does.
    let rs: Vec<(usize, usize)> = w
        .rsb
        .ranges()
        .map(|r| (*r.start(), *r.end()))
        .collect();
    WeightKey::Ranges(rs)
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct SigKey {
    final_w: WeightKey,
    trans: Vec<(Label, WeightKey, usize)>, // (label, weight, succ_sig_id)
}

/// Computes a canonical “behavior signature id” for the start state.
/// Two DWAs are equivalent (from the start) iff these ids are equal when computed
/// with the *same* interner table.
///
/// This is exact for acyclic deterministic machines under your eval semantics.
fn start_behavior_id(dwa: &DWA, interner: &mut HashMap<SigKey, usize>) -> Result<usize, DWAMinimizeError> {
    let n = dwa.states.len();
    if n == 0 {
        // empty machine: always outputs empty
        let key = SigKey { final_w: WeightKey::Empty, trans: vec![] };
        let id = *interner.entry(key).or_insert_with(|| interner.len());
        return Ok(id);
    }
    let start = dwa.body.start_state;
    if start >= n {
        return Err(DWAMinimizeError::StartOutOfBounds { start, n });
    }

    let order = topo_order(dwa)?;
    let mut sig_id = vec![0usize; n];

    for &u in order.iter().rev() {
        let fw = dwa.states[u].final_weight.clone().unwrap_or_else(Weight::zeros);
        let final_w = weight_key(&fw);

        let mut trans = Vec::with_capacity(dwa.states[u].transitions.len());
        for (lbl, &v) in &dwa.states[u].transitions {
            let w = dwa.states[u].trans_weights.get(lbl).cloned().unwrap_or_else(Weight::all);
            trans.push((*lbl, weight_key(&w), sig_id[v]));
        }

        // transitions are already sorted by BTreeMap iteration order, but keep it explicit:
        trans.sort_by_key(|(lbl, _, _)| *lbl);

        let key = SigKey { final_w, trans };
        let id = *interner.entry(key).or_insert_with(|| interner.len());
        sig_id[u] = id;
    }

    Ok(sig_id[start])
}

/// Strict minimization: merges states that are *already* equivalent (identical residual behavior).
fn minimize_strict_acyclic(input: &DWA) -> Result<DWA, DWAMinimizeError> {
    let n = input.states.len();
    if n == 0 {
        return Ok(input.clone());
    }
    let start = input.body.start_state;
    if start >= n {
        return Err(DWAMinimizeError::StartOutOfBounds { start, n });
    }

    let order = topo_order(input)?;
    let mut map: HashMap<SigKey, usize> = HashMap::new();
    let mut rep_of = vec![0usize; n];
    let mut reps: Vec<StateID> = Vec::new(); // representative old state for each new state

    // Build signatures bottom-up.
    for &u in order.iter().rev() {
        let fw = input.states[u].final_weight.clone().unwrap_or_else(Weight::zeros);
        let final_w = weight_key(&fw);

        let mut trans = Vec::with_capacity(input.states[u].transitions.len());
        for (lbl, &v) in &input.states[u].transitions {
            let w = input.states[u].trans_weights.get(lbl).cloned().unwrap_or_else(Weight::all);
            trans.push((*lbl, weight_key(&w), rep_of[v]));
        }
        trans.sort_by_key(|(lbl, _, _)| *lbl);

        let key = SigKey { final_w, trans };
        let id = *map.entry(key).or_insert_with(|| {
            let new_id = reps.len();
            reps.push(u);
            new_id
        });
        rep_of[u] = id;
    }

    // Build minimized automaton.
    let mut new_states = vec![DWAState::default(); reps.len()];
    for (new_id, &u) in reps.iter().enumerate() {
        // final
        new_states[new_id].final_weight = input.states[u].final_weight.clone();

        // transitions
        for (lbl, &v) in &input.states[u].transitions {
            let w = input.states[u].trans_weights.get(lbl).cloned().unwrap_or_else(Weight::all);
            new_states[new_id].transitions.insert(*lbl, rep_of[v]);
            new_states[new_id].trans_weights.insert(*lbl, w);
        }
    }

    let mut out = DWA {
        states: DWAStates(new_states),
        body: DWABody { start_state: rep_of[start] },
    };
    out = trim_unreachable(out)?;
    Ok(out)
}

/* ========================================================================================== */
/* == Step 3: exact merge search (rollback DSU + branch-and-bound) =========================== */
/* ========================================================================================== */

#[derive(Clone, Debug)]
struct BitSet {
    words: Vec<u64>,
}

impl BitSet {
    fn new(nbits: usize) -> Self {
        Self { words: vec![0u64; (nbits + 63) / 64] }
    }
    fn set(&mut self, i: usize) {
        self.words[i / 64] |= 1u64 << (i % 64);
    }
    fn or_assign(&mut self, other: &BitSet) {
        for (a, b) in self.words.iter_mut().zip(other.words.iter()) {
            *a |= *b;
        }
    }
    fn intersects(&self, other: &BitSet) -> bool {
        self.words.iter().zip(other.words.iter()).any(|(a, b)| (*a & *b) != 0)
    }
}

#[derive(Clone)]
enum Hist {
    Union {
        child: usize,
        parent_before: usize,
        root: usize,
        size_before: usize,
        members_before: BitSet,
        blocked_before: BitSet,
        final_before: Weight,
        trans_before: BTreeMap<Label, (StateID, Weight)>,
    },
    Block {
        a: usize,
        blocked_a_before: BitSet,
        b: usize,
        blocked_b_before: BitSet,
    },
}

struct RBMerge {
    n: usize,
    parent: Vec<usize>,
    size: Vec<usize>,
    members: Vec<BitSet>,
    blocked: Vec<BitSet>,
    final_w: Vec<Weight>,
    trans: Vec<BTreeMap<Label, (StateID, Weight)>>,
    hist: Vec<Hist>,
}

impl RBMerge {
    fn new(base: &DWA) -> Result<Self, DWAMinimizeError> {
        let n = base.states.len();
        let order = topo_order(base)?; // ensure acyclic

        // Precompute reachability bitsets for “cannot merge ancestors/descendants”.
        let mut reach = vec![BitSet::new(n); n];
        for &u in order.iter().rev() {
            for &v in base.states[u].transitions.values() {
                if v >= n {
                    continue;
                }
                reach[u].set(v);
                let rv = reach[v].clone();
                reach[u].or_assign(&rv);
            }
        }

        let mut rev_reach = vec![BitSet::new(n); n];
        for u in 0..n {
            for v in 0..n {
                if reach[u].intersects(&{
                    let mut bs = BitSet::new(n);
                    bs.set(v);
                    bs
                }) && reach[u].words[v / 64] & (1u64 << (v % 64)) != 0
                {
                    rev_reach[v].set(u);
                }
            }
        }

        let mut parent = (0..n).collect::<Vec<_>>();
        let mut size = vec![1usize; n];

        let mut members = vec![BitSet::new(n); n];
        for i in 0..n {
            members[i].set(i);
        }

        // Initial blocked set = (reachable either way) + self.
        let mut blocked = vec![BitSet::new(n); n];
        for i in 0..n {
            blocked[i].set(i);
            blocked[i].or_assign(&reach[i]);
            blocked[i].or_assign(&rev_reach[i]);
        }

        let mut final_w = vec![Weight::zeros(); n];
        let mut trans = vec![BTreeMap::<Label, (StateID, Weight)>::new(); n];
        for i in 0..n {
            if let Some(fw) = &base.states[i].final_weight {
                final_w[i] = fw.clone();
            }
            for (lbl, &to) in &base.states[i].transitions {
                let w = base.states[i].trans_weights.get(lbl).cloned().unwrap_or_else(Weight::all);
                trans[i].insert(*lbl, (to, w));
            }
        }

        Ok(Self {
            n,
            parent,
            size,
            members,
            blocked,
            final_w,
            trans,
            hist: vec![],
        })
    }

    fn snapshot(&self) -> usize {
        self.hist.len()
    }

    fn rollback(&mut self, snap: usize) {
        while self.hist.len() > snap {
            match self.hist.pop().unwrap() {
                Hist::Union {
                    child,
                    parent_before,
                    root,
                    size_before,
                    members_before,
                    blocked_before,
                    final_before,
                    trans_before,
                } => {
                    self.parent[child] = parent_before;
                    self.size[root] = size_before;
                    self.members[root] = members_before;
                    self.blocked[root] = blocked_before;
                    self.final_w[root] = final_before;
                    self.trans[root] = trans_before;
                }
                Hist::Block { a, blocked_a_before, b, blocked_b_before } => {
                    self.blocked[a] = blocked_a_before;
                    self.blocked[b] = blocked_b_before;
                }
            }
        }
    }

    fn find(&self, mut x: usize) -> usize {
        while self.parent[x] != x {
            x = self.parent[x];
        }
        x
    }

    fn roots(&self) -> Vec<usize> {
        (0..self.n).filter(|&i| self.parent[i] == i).collect()
    }

    fn can_merge_roots(&self, ra: usize, rb: usize) -> bool {
        if ra == rb {
            return true;
        }
        // symmetric check: members(ra) ∩ blocked(rb) = ∅ and vice versa
        if self.members[ra].intersects(&self.blocked[rb]) {
            return false;
        }
        if self.members[rb].intersects(&self.blocked[ra]) {
            return false;
        }
        true
    }

    fn add_block(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        let blocked_a_before = self.blocked[ra].clone();
        let blocked_b_before = self.blocked[rb].clone();
        self.hist.push(Hist::Block { a: ra, blocked_a_before, b: rb, blocked_b_before });

        let mb = self.members[rb].clone();
        let ma = self.members[ra].clone();
        self.blocked[ra].or_assign(&mb);
        self.blocked[rb].or_assign(&ma);
    }

    fn union_roots(&mut self, ra: usize, rb: usize, todo: &mut Vec<(usize, usize)>) -> bool {
        let ra = self.find(ra);
        let rb = self.find(rb);
        if ra == rb {
            return true;
        }
        if !self.can_merge_roots(ra, rb) {
            return false;
        }

        // union-by-size: root = a, child = b
        let (a, b) = if self.size[ra] >= self.size[rb] { (ra, rb) } else { (rb, ra) };

        // Any common label forces merging the targets to keep determinism.
        for (lbl, (tb, _)) in self.trans[b].iter() {
            if let Some((ta, _)) = self.trans[a].get(lbl) {
                todo.push((*ta, *tb));
            }
        }

        // Save rollback info for root `a` and child `b`.
        self.hist.push(Hist::Union {
            child: b,
            parent_before: self.parent[b],
            root: a,
            size_before: self.size[a],
            members_before: self.members[a].clone(),
            blocked_before: self.blocked[a].clone(),
            final_before: self.final_w[a].clone(),
            trans_before: self.trans[a].clone(),
        });

        // Perform union.
        self.parent[b] = a;
        self.size[a] += self.size[b];
        self.members[a].or_assign(&self.members[b]);
        self.blocked[a].or_assign(&self.blocked[b]);

        self.final_w[a] |= &self.final_w[b];

        // Merge transition maps: union weights; keep one target (closure merges the targets anyway).
        let trans_b = self.trans[b].clone();
        for (lbl, (tb, wb)) in trans_b.into_iter() {
            self.trans[a]
                .entry(lbl)
                .and_modify(|(_ta, wa)| {
                    *wa |= &wb;
                })
                .or_insert((tb, wb));
        }

        true
    }

    fn merge_with_closure(&mut self, a: usize, b: usize) -> bool {
        let mut todo = vec![(a, b)];
        while let Some((x, y)) = todo.pop() {
            let rx = self.find(x);
            let ry = self.find(y);
            if rx == ry {
                continue;
            }
            if !self.union_roots(rx, ry, &mut todo) {
                return false;
            }
        }
        true
    }

    fn build_quotient_dwa(&self, start_old: usize) -> DWA {
        let roots = self.roots();
        let mut root_to_new = vec![None; self.n];
        for (nid, &r) in roots.iter().enumerate() {
            root_to_new[r] = Some(nid);
        }

        let mut states = vec![DWAState::default(); roots.len()];
        for (nid, &r) in roots.iter().enumerate() {
            let fw = self.final_w[r].clone();
            if !fw.is_empty() {
                states[nid].final_weight = Some(fw);
            }

            for (lbl, (to_old, w)) in &self.trans[r] {
                if w.is_empty() {
                    continue;
                }
                let to_root = self.find(*to_old);
                let to_new = root_to_new[to_root].unwrap();
                states[nid].transitions.insert(*lbl, to_new);
                states[nid].trans_weights.insert(*lbl, w.clone());
            }
        }

        let start_root = self.find(start_old);
        let start_new = root_to_new[start_root].unwrap();

        DWA {
            states: DWAStates(states),
            body: DWABody { start_state: start_new },
        }
    }
}

/// Exact search driver.
fn minimize_exact_by_merging(base: &DWA, step_limit: Option<u64>) -> Result<DWA, DWAMinimizeError> {
    let mut dsu = RBMerge::new(base)?;
    let mut interner: HashMap<SigKey, usize> = HashMap::new();
    let base_id = start_behavior_id(base, &mut interner)?;

    // Initial best = base (already correct).
    let mut best = base.clone();
    let mut best_n = best.states.len();

    let mut steps: u64 = 0;

    fn greedy_clique_lower_bound(dsu: &RBMerge, roots: &[usize]) -> usize {
        // Greedy clique in the incompatibility graph implied by current blocked sets.
        // This is only a lower bound, but cheap and sometimes prunes.
        let mut verts = roots.to_vec();
        // sort by “blocked degree” approximation
        verts.sort_by_key(|&r| {
            // degree ≈ number of 1 bits in blocked[r] intersect roots
            // we compute a cheap proxy: sum popcount(blocked words)
            dsu.blocked[r].words.iter().map(|w| w.count_ones() as usize).sum::<usize>()
        });
        verts.reverse();

        let mut clique: Vec<usize> = vec![];
        'outer: for &v in &verts {
            for &u in &clique {
                // Need u incompatible with v, i.e., u ∈ blocked[v] or v ∈ blocked[u]
                // We test via members/blocked intersection.
                if dsu.can_merge_roots(u, v) {
                    continue 'outer;
                }
            }
            clique.push(v);
        }
        clique.len().max(1)
    }

    fn dfs(
        base: &DWA,
        dsu: &mut RBMerge,
        interner: &mut HashMap<SigKey, usize>,
        base_id: usize,
        best: &mut DWA,
        best_n: &mut usize,
        steps: &mut u64,
        step_limit: Option<u64>,
    ) -> Result<(), DWAMinimizeError> {
        *steps += 1;
        if let Some(limit) = step_limit {
            if *steps > limit {
                return Ok(());
            }
        }

        let roots = dsu.roots();

        // Prune if even an optimistic lower bound can't beat best.
        let lb = greedy_clique_lower_bound(dsu, &roots);
        if lb >= *best_n {
            return Ok(());
        }

        // Find a mergeable pair to branch on.
        let mut pair: Option<(usize, usize)> = None;
        'find: for i in 0..roots.len() {
            for j in (i + 1)..roots.len() {
                let a = roots[i];
                let b = roots[j];
                if dsu.can_merge_roots(a, b) {
                    pair = Some((a, b));
                    break 'find;
                }
            }
        }

        if let Some((a, b)) = pair {
            // Branch 1: merge (with closure).
            let snap = dsu.snapshot();
            if dsu.merge_with_closure(a, b) {
                dfs(base, dsu, interner, base_id, best, best_n, steps, step_limit)?;
            }
            dsu.rollback(snap);

            // Branch 2: forbid merging these components.
            let snap = dsu.snapshot();
            dsu.add_block(a, b);
            dfs(base, dsu, interner, base_id, best, best_n, steps, step_limit)?;
            dsu.rollback(snap);

            return Ok(());
        }

        // No mergeable pairs left: evaluate this partition.
        let cand = dsu.build_quotient_dwa(base.body.start_state);
        let cand = trim_unreachable(cand)?;
        let cand_n = cand.states.len();
        if cand_n >= *best_n {
            return Ok(());
        }

        let cand_id = start_behavior_id(&cand, interner)?;
        if cand_id == base_id {
            *best = cand;
            *best_n = cand_n;
        }

        Ok(())
    }

    dfs(
        base,
        &mut dsu,
        &mut interner,
        base_id,
        &mut best,
        &mut best_n,
        &mut steps,
        step_limit,
    )?;

    Ok(best)
}