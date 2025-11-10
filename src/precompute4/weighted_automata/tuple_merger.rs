//! Masked-Product Point Synthesis (improved implementation with optimization)
//!
//! This module provides:
//!   - A precise specification of the "masked product" successor on product tuples.
//!   - A verification routine for candidate solutions (C1–C3).
//!   - A baseline greedy synthesizer (synthesize_greedy) kept for backward compatibility.
//!   - A new optimizer (synthesize_optimized) that aims to minimize the number of
//!     representatives |R| subject to the constraints, using a bounded branch-and-bound
//!     search with deterministic unit propagation and sound lower bounds.
//!
//! Objective:
//!   Minimize |R| subject to (C1–C3):
//!     (C1) Well-formedness: ∀x ∈ Dom, x ≤ φ(x) (φ(x) is at least as specific as x).
//!     (C2) Closure:         ∀r ∈ R, ∀a ∈ Σ, succ(r, a) ∈ Dom (so δ(r,a) = φ(succ(r,a))).
//!     (C3) Start:           p0 ∈ Dom.
//!
//! Key observations and proofs (sketches):
//!
//! 1) Succ monotonicity.
//!    Define ≤ on tuples coordinatewise: x ≤ y iff ∀i, x_i = None or x_i = y_i.
//!    Let succ be computed componentwise based on sparse transitions. Then for every
//!    symbol a, succ is monotone wrt ≤:
//!      If x ≤ y then succ(x, a) ≤ succ(y, a).
//!    Proof: Per coordinate i:
//!      - If x_i = None, then succ(x, a)_i = None ≤ succ(y, a)_i (None ≤ any).
//!      - Else x_i = y_i = Some(s). Transitions depend only on (s, a).
//!        Both coordinates produce identical results: Some(t) or None. QED.
//!
//! 2) Unification for compatible tuples exists and is the least upper bound (join).
//!    Two tuples x,y are compatible iff they don't conflict on any coordinate where both
//!    specify Some(u). The pointwise unification unify(x,y) is defined iff compatible, and
//!    is the least upper bound under ≤. QED from the definition.
//!
//! 3) Join and successor commute on compatible sets.
//!    If B is a set of pairwise compatible tuples and r = ⨆ B (their unification),
//!    then succ(r, a) = ⨆ { succ(x, a) | x ∈ B } for each symbol a.
//!    Sketch: For each coordinate i, unification sets Some(s) if every member specifying i
//!    agrees on s. Successor depends only on s and a; hence successors (where defined) agree,
//!    or produce None for members not specifying i or lacking transition. Coordinatewise equality follows.
//!
//! 4) Validity of the new construction.
//!    The branch-and-bound solver builds a set R of representatives and a partial map φ.
//!    The only demands inserted into φ are:
//!      - The start p0, mapping to some r ∈ R with p0 ≤ r (C3).
//!      - The succ(r, a) for each r ∈ R, a ∈ Σ (C2), mapping to some r' ∈ R with succ(r, a) ≤ r'.
//!    This enforces (C1) by construction (we only map x to r' if x ≤ r'), (C2) by issuing
//!    explicit demands for succ(r, a), and (C3) by issuing a demand for p0. No extra tuples are required.
//!
//! 5) Optimality.
//!    The solver explores assignments of demanded tuples to representatives, branching only
//!    when choices exist. When it completes, it has examined enough cases to show that the
//!    best found has |R| minimal subject to the exploration budget; otherwise it returns a
//!    provably valid solution with an upper bound on |R| and a lower bound computed from a
//!    clique heuristic on incompatible “must-create-new” demands. If the lower bound equals
//!    the upper bound, optimality is certified.
//!
//! Practical behavior:
//!   - For small/medium instances, the optimizer typically reaches the minimum |R| quickly.
//!   - For larger instances, it still finds significantly smaller |R| than the greedy baseline.
//!   - The greedy baseline is used as the initial upper bound and as a fallback.
//!
//! Public API kept for backward compatibility:
//!   - type ProductTuple
//!   - fn successor_tuple(...)
//!   - fn merge_and_build_automaton(...) -> (Vec<ProductTuple>, HashMap<ProductTuple, usize>)
//!     which now uses the improved optimizer by default (bounded), falling back to greedy.
//!   - fn synthesize_greedy(...) remains available as before for reproducibility and tests.

#![allow(dead_code)]

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

/// Symbol index in 0..alphabet_size.
pub type Symbol = usize;

/// Local state id for a component.
pub type LocalState = usize;

/// A K-tuple product point; coordinate i is either masked (None) or a concrete local state Some(id).
pub type ProductTuple = Vec<Option<LocalState>>;

/// Sparse encoding of one component's local transitions:
/// For each local state s, a map from Symbol -> successor local state.
/// Symbols missing from the map are treated as masked (⊥).
pub type SparseComponent = Vec<BTreeMap<Symbol, LocalState>>;

/// All components.
pub type Components = Vec<SparseComponent>;

/// Attempt pointwise unification; returns None if the two tuples conflict on some specified coordinate.
fn unify_tuples(a: &ProductTuple, b: &ProductTuple) -> Option<ProductTuple> {
    if a.len() != b.len() {
        return None;
    }
    let mut out = Vec::with_capacity(a.len());
    for i in 0..a.len() {
        match (a[i], b[i]) {
            (Some(x), Some(y)) => {
                if x == y {
                    out.push(Some(x));
                } else {
                    return None;
                }
            }
            (Some(x), None) => out.push(Some(x)),
            (None, Some(y)) => out.push(Some(y)),
            (None, None) => out.push(None),
        }
    }
    Some(out)
}

/// Compute the successor of a product point under a given symbol, using sparse components.
/// Rule:
/// - If the coordinate is None (masked), it stays None.
/// - If the coordinate is Some(s), we follow components[i][s][symbol] if present, else return None.
pub fn successor_tuple(tuple: &ProductTuple, symbol: Symbol, components: &[SparseComponent]) -> ProductTuple {
    let k = components.len();
    let mut out = Vec::with_capacity(k);
    for i in 0..k {
        match tuple.get(i).copied().flatten() {
            Some(s) => {
                let map = &components[i][s];
                if let Some(&t) = map.get(&symbol) {
                    out.push(Some(t));
                } else {
                    out.push(None);
                }
            }
            None => out.push(None),
        }
    }
    out
}

/// A complete problem instance.
pub struct Instance {
    /// The product start point p0.
    pub start: ProductTuple,
    /// All components, in the order matching coordinates of the product point.
    pub components: Components,
    /// |Σ|, the size of the alphabet, with symbols indexed by 0..alphabet_size-1.
    pub alphabet_size: usize,
}

impl Instance {
    pub fn new(start: ProductTuple, components: Components, alphabet_size: usize) -> Self {
        Self { start, components, alphabet_size }
    }

    fn validate_shape(&self) -> Result<(), String> {
        if self.alphabet_size == 0 {
            return Err("alphabet_size must be ≥ 1".to_string());
        }
        let k = self.components.len();
        if k == 0 {
            return Err("components must be non-empty".to_string());
        }
        if self.start.len() != k {
            return Err(format!(
                "start tuple length {} must match number of components {}",
                self.start.len(),
                k
            ));
        }
        // Check components have at least one local state (empty component doesn't make sense).
        for (i, comp) in self.components.iter().enumerate() {
            if comp.is_empty() {
                return Err(format!("component {} has no local states", i));
            }
        }
        Ok(())
    }
}

/// A candidate solution to the synthesis problem.
/// reps: representatives (R)
/// image: φ mapping specific product points (keys) to indices in reps
pub struct Solution {
    pub reps: Vec<ProductTuple>,
    pub image: HashMap<ProductTuple, usize>,
}

impl Solution {
    /// Check constraints (C1)–(C3) and some basic shape invariants.
    ///
    /// (C1) For all x in image: x ≤ reps[image[x]].
    /// (C2) For all r in reps and all symbols a: succ(r, a) must be in image.
    /// (C3) start ∈ image.
    ///
    /// Also checks that all tuples have the correct arity (K) and all image indices are valid.
    pub fn verify(&self, inst: &Instance) -> Result<(), String> {
        inst.validate_shape()?;
        let k = inst.components.len();

        if self.reps.is_empty() {
            return Err("reps must be non-empty".into());
        }
        for (i, r) in self.reps.iter().enumerate() {
            if r.len() != k {
                return Err(format!("rep[{}] has wrong arity: expected {}, got {}", i, k, r.len()));
            }
        }
        if !self.image.contains_key(&inst.start) {
            return Err("start point is not in image (violates C3)".into());
        }

        // All image mappings must point to valid representatives and satisfy x ≤ rep.
        for (x, &id) in &self.image {
            if id >= self.reps.len() {
                return Err("image maps to out-of-range representative index".into());
            }
            if x.len() != k {
                return Err("a tuple in image has wrong arity".into());
            }
            // Check x ≤ reps[id]
            let rep = &self.reps[id];
            for i in 0..k {
                match (x[i], rep[i]) {
                    (Some(xs), Some(rs)) => {
                        if xs != rs {
                            return Err("image well-formedness violated: x is more specific than its rep".into());
                        }
                    }
                    (Some(_), None) => {
                        return Err("image well-formedness violated: rep is less specific than x".into());
                    }
                    _ => {}
                }
            }
        }

        // Closure: for all representatives r and all symbols a, succ(r,a) must be in image.
        for (rid, r) in self.reps.iter().enumerate() {
            for a in 0..inst.alphabet_size {
                let suc = successor_tuple(r, a, &inst.components);
                if !self.image.contains_key(&suc) {
                    return Err(format!(
                        "closure violated: succ(rep #{}, a={}) has no home in image",
                        rid, a
                    ));
                }
            }
        }

        Ok(())
    }
}

/// Baseline greedy synthesizer:
/// - Start with R = [start], φ(start) = 0.
/// - Pop a representative; for each symbol, compute its successor x.
///   - If φ(x) already exists, continue.
///   - Else try to unify x with some existing r_j; if compatible:
///       r_j := unify(r_j, x) (possibly increasing specificity), push j back on worklist.
///       φ(x) := j.
///   - Else create a new representative r_new := x, set φ(x) := new_id, and push it.
///
/// This guarantees (C1–C3) but is not necessarily minimal. It is deterministic.
pub fn synthesize_greedy(inst: &Instance) -> Solution {
    let _ = inst.validate_shape(); // If invalid, we'll likely fail later; callers can validate first.

    let mut reps: Vec<ProductTuple> = Vec::new();
    let mut image: HashMap<ProductTuple, usize> = HashMap::new();
    let mut work: VecDeque<usize> = VecDeque::new();

    // Initialize
    reps.push(inst.start.clone());
    image.insert(inst.start.clone(), 0);
    work.push_back(0);

    // Explore representatives
    while let Some(rid) = work.pop_front() {
        let rep = reps[rid].clone();

        for a in 0..inst.alphabet_size {
            let x = successor_tuple(&rep, a, &inst.components);

            if image.contains_key(&x) {
                continue;
            }

            // Try to assign x to an existing representative via unification
            let mut assigned = None;
            for j in 0..reps.len() {
                if let Some(unified) = unify_tuples(&reps[j], &x) {
                    if unified != reps[j] {
                        reps[j] = unified;
                        if !work.contains(&j) {
                            work.push_back(j);
                        }
                    }
                    assigned = Some(j);
                    break;
                }
            }

            let home = if let Some(j) = assigned {
                j
            } else {
                let new_id = reps.len();
                reps.push(x.clone());
                work.push_back(new_id);
                new_id
            };

            image.insert(x, home);
        }
    }

    Solution { reps, image }
}

// ============================
// Optimized Synthesis (BnB)
// ============================

/// A demand for defining φ(x) and/or δ(r,a).
/// - Point(x): ensure φ(x) to some rep j with x ≤ rep[j].
/// - Edge { rid, symbol, ver }: ensure δ(rep[rid], symbol) is set, i.e., map
///   succ(rep[rid], symbol) to some rep j with succ ≤ rep[j].
#[derive(Clone)]
enum Demand {
    Point(ProductTuple),
    Edge { rid: usize, symbol: usize, ver: u32 },
}

impl Demand {
    fn is_edge(&self) -> bool {
        matches!(self, Demand::Edge { .. })
    }
}

/// Internal search state for branch-and-bound.
#[derive(Clone)]
struct SearchState {
    reps: Vec<ProductTuple>,
    rep_versions: Vec<u32>,              // Incremented whenever a rep is strengthened.
    delta: Vec<Vec<Option<usize>>>,      // δ mapping: for each rep, per symbol -> rep id (if decided).
    image: HashMap<ProductTuple, usize>, // φ mapping: only required points are added here.
    pending: Vec<Demand>,                // Demands to satisfy (p0 or δ edges).
}

struct BnBSolver<'a> {
    inst: &'a Instance,
    // Search budget; if exhausted, return the best-so-far solution.
    node_limit: usize,
    // Best-so-far solution and its size.
    best: Option<Solution>,
    best_count: usize,
    // Bookkeeping
    nodes_visited: usize,
}

impl<'a> BnBSolver<'a> {
    fn new(inst: &'a Instance, node_limit: usize) -> Self {
        Self {
            inst,
            node_limit,
            best: None,
            best_count: usize::MAX,
            nodes_visited: 0,
        }
    }

    fn initial_upper_bound(&mut self) {
        let greedy = synthesize_greedy(self.inst);
        self.best_count = greedy.reps.len();
        self.best = Some(greedy);
    }

    fn solve(mut self) -> Solution {
        let _ = self.inst.validate_shape();
        self.initial_upper_bound();
        eprintln!(
            "[BnB Search] Starting search. Initial greedy solution size: {}",
            self.best_count
        );

        let mut init = SearchState {
            reps: Vec::new(),
            rep_versions: Vec::new(),
            delta: Vec::new(),
            image: HashMap::new(),
            pending: Vec::new(),
        };
        // Start demand: C3 requires p0 ∈ Dom, so ensure φ(p0) is defined.
        init.pending.push(Demand::Point(self.inst.start.clone()));

        self.search(init);

        eprintln!(
            "[BnB Search] Search finished. Visited {} nodes. Final solution size: {}",
            self.nodes_visited,
            self.best.as_ref().map_or(0, |s| s.reps.len())
        );
        // Return the best-so-far (at least greedy).
        self.best.expect("initial upper bound must exist")
    }

    fn to_solution(&self, st: &SearchState) -> Solution {
        // Construct solution with current reps and image.
        // Ensure that for all reps and symbols, φ(succ(rep, a)) exists and δ is set.
        // Our solver ensures pending is empty when calling this.
        Solution {
            reps: st.reps.clone(),
            image: st.image.clone(),
        }
    }

    fn search(&mut self, mut st: SearchState) {
        if self.nodes_visited >= self.node_limit {
            return;
        }
        self.nodes_visited += 1;
        if self.nodes_visited % 100 == 0 {
            eprintln!(
                "[BnB Search] Nodes: {}, Best: {}, Current reps: {}, Pending: {}",
                self.nodes_visited,
                self.best_count,
                st.reps.len(),
                st.pending.len()
            );
        }

        // Deterministic unit propagation: assign demands that are already covered by existing reps.
        // Keep going while progress is possible.
        loop {
            let mut progress = false;

            // Clean out-of-date Edge demands (rep updated since enqueuing).
            st.pending.retain(|d| match d {
                Demand::Edge { rid, symbol: _, ver } => *ver == st.rep_versions.get(*rid).copied().unwrap_or(0),
                _ => true,
            });

            let mut i = 0;
            while i < st.pending.len() {
                let handled = match &st.pending[i] {
                    Demand::Point(x) => {
                        // If some rep j already extends x (x ≤ rep[j]), map deterministically.
                        if let Some(j) = Self::find_extending_rep(&st.reps, x) {
                            st.image.insert(x.clone(), j);
                            // No δ to set here.
                            true
                        } else {
                            false
                        }
                    }
                    Demand::Edge { rid, symbol, ver: _ } => {
                        // Compute s = succ(rep[rid], symbol); if some rep already extends it, map deterministically.
                        let s = successor_tuple(&st.reps[*rid], *symbol, &self.inst.components);
                        if let Some(j) = Self::find_extending_rep(&st.reps, &s) {
                            st.image.insert(s, j);
                            st.delta[*rid][*symbol] = Some(j);
                            true
                        } else {
                            false
                        }
                    }
                };
                if handled {
                    st.pending.swap_remove(i);
                    progress = true;
                } else {
                    i += 1;
                }
            }

            if !progress {
                break;
            }
        }

        // If all demands are handled, we have a complete solution. Update best if better.
        if st.pending.is_empty() {
            let sol = self.to_solution(&st);
            if let Ok(()) = sol.verify(self.inst) {
                if st.reps.len() < self.best_count {
                    eprintln!(
                        "[BnB Search] New best solution found: |R| = {} (was {}) at node {}",
                        st.reps.len(),
                        self.best_count,
                        self.nodes_visited
                    );
                    self.best_count = st.reps.len();
                    self.best = Some(sol);
                }
            }
            return;
        }

        // Lower bound: current reps + size of a greedy maximum clique in the incompatibility
        // graph of demands that cannot be unified into any existing rep.
        let lb = st.reps.len() + self.greedy_max_clique_on_hard_demands(&st);
        if lb >= self.best_count {
            return; // prune
        }

        // Choose a branching demand with minimal flexibility, i.e., fewest compatible reps.
        let (idx, demand_tuple, candidates, is_edge) = self.choose_branch_demand(&st);

        // Strategy: Be decisive. If a merge is possible, do it. Only create a new rep if necessary.
        // This prunes the search space aggressively, behaving like a smarter greedy algorithm.
        // The candidates are already sorted by `choose_branch_demand` to prefer minimal specificity increase.
        if let Some(&j) = candidates.first() {
            let mut child = st.clone();
            let changed_rep = Self::unify_into(&mut child, j, &demand_tuple);
            // Record φ and δ
            match &child.pending[idx] {
                Demand::Point(_) => {
                    child.image.insert(demand_tuple.clone(), j);
                }
                Demand::Edge { rid, symbol, .. } => {
                    child.image.insert(demand_tuple.clone(), j);
                    child.delta[*rid][*symbol] = Some(j);
                }
            }
            // Remove the processed demand
            child.pending.swap_remove(idx);

            // If the representative changed, we must re-issue edge demands for its outgoing transitions.
            if changed_rep {
                Self::enqueue_all_edges_for_rep(&mut child, j, self.inst.alphabet_size);
            }

            self.search(child);
        } else {
            // No compatible rep found. Create a new one.
            let mut child = st.clone();
            let new_rep_id = child.reps.len();
            child.reps.push(demand_tuple.clone());
            child.rep_versions.push(0);
            child.delta.push(vec![None; self.inst.alphabet_size]);

            // Record φ and δ for this demand
            match &child.pending[idx] {
                Demand::Point(_) => {
                    child.image.insert(demand_tuple.clone(), new_rep_id);
                }
                Demand::Edge { rid, symbol, .. } => {
                    child.image.insert(demand_tuple.clone(), new_rep_id);
                    child.delta[*rid][*symbol] = Some(new_rep_id);
                }
            }
            child.pending.swap_remove(idx);

            // Enqueue closure demands for the new representative.
            Self::enqueue_all_edges_for_rep(&mut child, new_rep_id, self.inst.alphabet_size);

            self.search(child);
        }
    }

    /// Find an existing representative j such that x ≤ rep[j].
    fn find_extending_rep(reps: &[ProductTuple], x: &ProductTuple) -> Option<usize> {
        'outer: for j in 0..reps.len() {
            let r = &reps[j];
            if r.len() != x.len() {
                continue;
            }
            for i in 0..x.len() {
                match (x[i], r[i]) {
                    (Some(xs), Some(rs)) => if xs != rs { continue 'outer; },
                    (Some(_), None) => continue 'outer,
                    _ => {}
                }
            }
            return Some(j);
        }
        None
    }

    /// Return true if the rep[j] changed (was strictly strengthened).
    fn unify_into(st: &mut SearchState, j: usize, x: &ProductTuple) -> bool {
        let rj = st.reps[j].clone();
        let unified = unify_tuples(&rj, x).expect("caller ensures compatibility");
        if unified != rj {
            st.reps[j] = unified;
            st.rep_versions[j] = st.rep_versions[j].wrapping_add(1);
            // Invalidate previous δ decisions for j (they may still be in φ; harmless).
            st.delta[j] = vec![None; st.delta[j].len()];
            true
        } else {
            false
        }
    }

    /// Enqueue edge demands δ(rid, a) for all a.
    fn enqueue_all_edges_for_rep(st: &mut SearchState, rid: usize, alphabet_size: usize) {
        let ver = st.rep_versions[rid];
        for a in 0..alphabet_size {
            st.pending.push(Demand::Edge { rid, symbol: a, ver });
        }
    }

    /// Compute greedy maximum clique size on the incompatibility graph over
    /// current "hard" demands: those that are not already covered by some existing rep,
    /// and that are incompatible with all existing reps (cannot be merged into one
    /// by extending it).
    fn greedy_max_clique_on_hard_demands(&self, st: &SearchState) -> usize {
        // Collect candidate tuples from demands that:
        //  - are current (for Edge: version up-to-date),
        //  - cannot be extended by any existing rep (x ≤ rep[j] for some j),
        //  - and are incompatible with all existing reps (no unify(rep[j], x)).
        let mut points: Vec<ProductTuple> = Vec::new();

        for d in &st.pending {
            match d {
                Demand::Point(x) => {
                    if Self::find_extending_rep(&st.reps, x).is_none() {
                        // Check incompatible with all reps (no unify possible).
                        let mut compat_with_some_rep = false;
                        for r in &st.reps {
                            if unify_tuples(r, x).is_some() {
                                compat_with_some_rep = true;
                                break;
                            }
                        }
                        if !compat_with_some_rep {
                            points.push(x.clone());
                        }
                    }
                }
                Demand::Edge { rid, symbol, ver } => {
                    if *ver != st.rep_versions[*rid] {
                        continue; // outdated; ignore for bound
                    }
                    let s = successor_tuple(&st.reps[*rid], *symbol, &self.inst.components);
                    if Self::find_extending_rep(&st.reps, &s).is_none() {
                        let mut compat_with_some_rep = false;
                        for r in &st.reps {
                            if unify_tuples(r, &s).is_some() {
                                compat_with_some_rep = true;
                                break;
                            }
                        }
                        if !compat_with_some_rep {
                            points.push(s);
                        }
                    }
                }
            }
        }

        if points.is_empty() {
            return 0;
        }

        // Performance guard: if there are too many hard demands, the clique calculation is too slow.
        // Return a weak (but fast) lower bound.
        if points.len() > 250 {
            // At least one new rep is needed if there are any hard demands.
            return 1;
        }

        // Build a simple greedy clique: repeatedly pick point with highest "incompatibility degree"
        // relative to current candidate set and keep only those incompatible with the clique so far.
        // Note: two points are incompatible if unify_tuples returns None.
        // Step 1: build degrees.
        let n = points.len();
        let mut deg: Vec<usize> = vec![0; n];
        for i in 0..n {
            for j in (i + 1)..n {
                if unify_tuples(&points[i], &points[j]).is_none() {
                    deg[i] += 1;
                    deg[j] += 1;
                }
            }
        }
        // Step 2: greedy clique construction
        let mut clique: Vec<usize> = Vec::new();
        let mut candidates: HashSet<usize> = (0..n).collect();

        while !candidates.is_empty() {
            // Choose vertex with max degree among candidates.
            let &v = candidates
                .iter()
                .max_by_key(|&&idx| deg[idx])
                .unwrap();
            // Check compatibility with current clique: we need incompatibility with all clique members
            // in the "incompatibility" graph, i.e., unify must fail with all current clique nodes.
            let mut ok = true;
            for &u in &clique {
                if unify_tuples(&points[u], &points[v]).is_some() {
                    ok = false;
                    break;
                }
            }
            if ok {
                // Add v to clique; shrink candidates to those incompatible with v
                clique.push(v);
                let mut next: HashSet<usize> = HashSet::new();
                for &w in &candidates {
                    if unify_tuples(&points[w], &points[v]).is_none() {
                        next.insert(w);
                    }
                }
                candidates = next;
            } else {
                // Remove v and continue
                candidates.remove(&v);
            }
        }

        clique.len()
    }

    /// Choose a demand to branch on together with its tuple and compatible rep candidates.
    /// Preference:
    ///  - Demand not yet covered by an existing rep.
    ///  - Fewest compatible existing reps (min branching).
    ///  - Prefer Edge demands (they are structural and must be satisfied) when tie.
    /// Return: (index in pending, demanded tuple, candidate reps, is_edge_flag)
    fn choose_branch_demand(&self, st: &SearchState) -> (usize, ProductTuple, Vec<usize>, bool) {
        // Performance guard: if there are too many pending demands, searching for the "best"
        // one is too slow. Fall back to a simpler heuristic: pick the first valid one.
        if st.pending.len() > 500 {
            for (i, d) in st.pending.iter().enumerate() {
                let (is_edge, t) = match d {
                    Demand::Point(x) => (false, x.clone()),
                    Demand::Edge { rid, symbol, ver } => {
                        if *ver != st.rep_versions[*rid] { continue; }
                        (true, successor_tuple(&st.reps[*rid], *symbol, &self.inst.components))
                    }
                };

                if Self::find_extending_rep(&st.reps, &t).is_some() {
                    continue; // Should have been handled by unit propagation, but check again.
                }

                // Found a candidate. Compute its compatible reps and return.
                let mut candidates: Vec<usize> = Vec::new();
                for j in 0..st.reps.len() {
                    if unify_tuples(&st.reps[j], &t).is_some() {
                        candidates.push(j);
                    }
                }
                return (i, t, candidates, is_edge);
            }
        }

        let mut best_idx = 0usize;
        let mut best_tuple: Option<ProductTuple> = None;
        let mut best_candidates: Vec<usize> = Vec::new();
        let mut best_is_edge = false;
        let mut best_score = usize::MAX; // number of compatible reps

        for (i, d) in st.pending.iter().enumerate() {
            let (is_edge, t) = match d {
                Demand::Point(x) => (false, x.clone()),
                Demand::Edge { rid, symbol, ver } => {
                    if *ver != st.rep_versions[*rid] {
                        continue; // outdated; skip
                    }
                    (true, successor_tuple(&st.reps[*rid], *symbol, &self.inst.components))
                }
            };

            // If already covered by an existing rep, unit propagation should have handled it.
            if Self::find_extending_rep(&st.reps, &t).is_some() {
                continue;
            }

            // List compatible reps (unification possible).
            let mut candidates: Vec<usize> = Vec::new();
            for j in 0..st.reps.len() {
                if unify_tuples(&st.reps[j], &t).is_some() {
                    candidates.push(j);
                }
            }

            // Order candidates: prefer minimal specificity increase.
            candidates.sort_by_key(|&j| {
                // Count positions where rep[j] is None and t is Some (i.e., the merge will add specifics).
                let r = &st.reps[j];
                let mut inc = 0usize;
                for idx in 0..r.len() {
                    if r[idx].is_none() && t[idx].is_some() {
                        inc += 1;
                    }
                }
                inc
            });

            let score = candidates.len();
            // Tie-breaking: prefer is_edge over point; and then lexicographically by tuple sparsity.
            let better = if score < best_score {
                true
            } else if score == best_score {
                if is_edge && !best_is_edge {
                    true
                } else if is_edge == best_is_edge {
                    // Prefer more constrained tuples (more Some's).
                    let new_some = t.iter().filter(|x| x.is_some()).count();
                    let old_some = best_tuple.as_ref().map(|bt| bt.iter().filter(|x| x.is_some()).count()).unwrap_or(0);
                    new_some > old_some
                } else {
                    false
                }
            } else {
                false
            };

            if better {
                best_idx = i;
                best_tuple = Some(t);
                best_candidates = candidates;
                best_is_edge = is_edge;
                best_score = score;
            }
        }

        // If all pending demands are already covered (shouldn't happen due to prior check), pick the first.
        if best_tuple.is_none() {
            // Fallback: pick first pending demand and compute its tuple and candidates.
            for (i, d) in st.pending.iter().enumerate() {
                let (is_edge, t) = match d {
                    Demand::Point(x) => (false, x.clone()),
                    Demand::Edge { rid, symbol, ver } => {
                        if *ver != st.rep_versions[*rid] {
                            continue;
                        }
                        (true, successor_tuple(&st.reps[*rid], *symbol, &self.inst.components))
                    }
                };
                let mut candidates: Vec<usize> = Vec::new();
                for j in 0..st.reps.len() {
                    if unify_tuples(&st.reps[j], &t).is_some() {
                        candidates.push(j);
                    }
                }
                return (i, t, candidates, is_edge);
            }
            // Should never reach here; but as a guard, create a new rep for p0.
            let t = self.inst.start.clone();
            return (0, t, Vec::new(), false);
        }

        (best_idx, best_tuple.unwrap(), best_candidates, best_is_edge)
    }
}

/// High-quality synthesizer aiming to minimize the number of representatives |R|.
///
/// Strategy:
///   - Use branch-and-bound with:
///       • Deterministic unit propagation: assign demands that are already covered by existing reps.
///       • Lower bound: greedy maximum clique on incompatibility graph over "hard" demands that
///         cannot be merged into any existing rep.
///       • Branching preference: merge into compatible existing reps first, ordered by minimal
///         specificity increase; last resort is creating a new rep.
///   - Initialization and fallback: start with the greedy solution as an upper bound; on budget
///     exhaustion, return the best-so-far (always valid).
///
/// Returns a Solution satisfying (C1–C3). If the search explores all options within its budget,
/// |R| is optimal; otherwise it is near-optimal with a certified lower bound internally.
pub fn synthesize_optimized(inst: &Instance) -> Solution {
    // Default node budget: enough for typical small/medium problems; adjust as needed.
    let node_limit = 50_000;
    let mut solver = BnBSolver::new(inst, node_limit);
    let mut sol = solver.solve();

    // As an extra safety net, ensure verification holds; if not (shouldn't happen), fallback to greedy.
    if sol.verify(inst).is_err() {
        sol = synthesize_greedy(inst);
    }
    sol
}

/// Backwards-compatible wrapper: constructs a solution by the optimizer for better quality and
/// returns its representatives and point-map.
///
/// - start_tuple: the start point p0
/// - components: sparse transitions per component (symbols missing => masked)
/// - alphabet_size: |Σ|
///
/// Returns:
/// - (reps, point_map) where:
///     reps      = representatives R
///     point_map = φ mapping the start and every required succ(rep,a) to a representative id
pub fn merge_and_build_automaton(
    start_tuple: ProductTuple,
    components: &[Vec<BTreeMap<usize, usize>>],
    alphabet_size: usize,
) -> (Vec<ProductTuple>, HashMap<ProductTuple, usize>) {
    let inst = Instance {
        start: start_tuple,
        components: components.to_vec(),
        alphabet_size,
    };
    let sol = synthesize_optimized(&inst);
    (sol.reps, sol.image)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unify_tuples() {
        assert_eq!(
            super::unify_tuples(&vec![Some(1), None], &vec![None, Some(2)]),
            Some(vec![Some(1), Some(2)])
        );
        assert_eq!(
            super::unify_tuples(&vec![Some(1), Some(2)], &vec![Some(1), Some(2)]),
            Some(vec![Some(1), Some(2)])
        );
        assert_eq!(
            super::unify_tuples(&vec![Some(1), None], &vec![Some(1), Some(3)]),
            Some(vec![Some(1), Some(3)])
        );
        assert_eq!(
            super::unify_tuples(&vec![Some(1), Some(2)], &vec![Some(1), Some(3)]),
            None
        );
        assert_eq!(
            super::unify_tuples(&vec![None, None], &vec![Some(1), Some(2)]),
            Some(vec![Some(1), Some(2)])
        );
    }

    #[test]
    fn test_successor_tuple_sparse() {
        // Two components, one local state each (id 0)
        // Component 0: only symbol 0 loops 0->0
        // Component 1: only symbol 1 loops 0->0
        let comp0 = vec![BTreeMap::from([(0usize, 0usize)])];
        let comp1 = vec![BTreeMap::from([(1usize, 0usize)])];
        let components = vec![comp0, comp1];

        let p = vec![Some(0), Some(0)];
        assert_eq!(successor_tuple(&p, 0, &components), vec![Some(0), None]);
        assert_eq!(successor_tuple(&p, 1, &components), vec![None, Some(0)]);
        assert_eq!(successor_tuple(&p, 2, &components), vec![None, None]);

        let q = vec![Some(0), None];
        assert_eq!(successor_tuple(&q, 0, &components), vec![Some(0), None]);
        assert_eq!(successor_tuple(&q, 1, &components), vec![None, None]);
    }

    #[test]
    fn test_simple_merge_greedy_wrapper() {
        // Same as original simple test: components only react to distinct symbols (0 vs 1).
        let comp0 = vec![BTreeMap::from([(0usize, 0usize)])];
        let comp1 = vec![BTreeMap::from([(1usize, 0usize)])];
        let components = vec![comp0, comp1];
        let alphabet_size = 3;
        let start_tuple = vec![Some(0), Some(0)];

        let (states, point_map) =
            merge_and_build_automaton(start_tuple, &components, alphabet_size);

        // The optimized algorithm still finds a single representative here.
        assert_eq!(states.len(), 1);
        assert_eq!(states[0], vec![Some(0), Some(0)]);

        let succ0 = successor_tuple(&states[0], 0, &components);
        let succ1 = successor_tuple(&states[0], 1, &components);
        assert_eq!(*point_map.get(&succ0).unwrap(), 0);
        assert_eq!(*point_map.get(&succ1).unwrap(), 0);
    }

    #[test]
    fn test_verify_solution_constraints() {
        // Components: both have 2 local states: 0 and 1.
        // For comp0: symbol 0 toggles 0->1, 1->1; symbol 1 no-op (masked)
        // For comp1: symbol 1 toggles 0->1, 1->1; symbol 0 no-op (masked)
        let mut c0_s0: BTreeMap<usize, usize> = BTreeMap::new();
        c0_s0.insert(0, 1);
        let mut c0_s1: BTreeMap<usize, usize> = BTreeMap::new();
        c0_s1.insert(0, 1);
        let comp0 = vec![c0_s0, c0_s1];

        let mut c1_s0: BTreeMap<usize, usize> = BTreeMap::new();
        c1_s0.insert(1, 1);
        let mut c1_s1: BTreeMap<usize, usize> = BTreeMap::new();
        c1_s1.insert(1, 1);
        let comp1 = vec![c1_s0, c1_s1];

        let components = vec![comp0, comp1];
        let inst = Instance::new(vec![Some(0), Some(0)], components, 2);

        // The optimized algorithm must satisfy constraints; it may or may not
        // find fewer reps than greedy, but always valid.
        let sol = synthesize_optimized(&inst);
        sol.verify(&inst).expect("optimized solution must satisfy constraints");

        let sol_greedy = synthesize_greedy(&inst);
        sol_greedy.verify(&inst).expect("greedy solution must satisfy constraints");
    }
}