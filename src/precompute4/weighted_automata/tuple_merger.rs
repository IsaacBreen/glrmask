//! Masked-Product Point Synthesis (self-contained spec and optimized greedy)
//!
//! Problem (informal):
//! 1. There are K components. Each component has a finite set of local states S_i plus a
//!    special "masked" value ⊥ (read: off/irrelevant).
//! 2. We have an alphabet of symbols Σ = {0,1,...,M-1}.
//! 3. For each component i and each local state s ∈ S_i, we define a partial transition
//!    τ_i(s, a) ∈ S_i on symbol a. If a transition does not exist, we treat it as masked (⊥).
//!    When a coordinate is already masked (⊥), it remains masked under any symbol.
//!
//! A product point is a K-tuple p = (p_0, ..., p_{K-1}) where each coordinate p_i ∈ S_i ∪ {⊥}.
//! A product successor is defined componentwise: succ(p, a)_i =
//!   - τ_i(p_i, a) if p_i ∈ S_i and the symbol a-transition exists;
//!   - ⊥ otherwise (including when p_i = ⊥).
//!
//! We write ⊥ as None in code; some local state x ∈ S_i is encoded as Some(x).
//!
//! Compatibility and Unification:
//! 1. Define a quasiorder ≤ on points: x ≤ y iff for all i, either x_i = ⊥ or x_i = y_i.
//!    Intuitively, y is at least as specific as x: whenever x specifies a concrete local state,
//!    y agrees. Otherwise x leaves the coordinate unspecified (⊥).
//! 2. Two points x,y are compatible iff they do not conflict on specified coordinates,
//!    i.e., ∀i: not (x_i = Some(u), y_i = Some(v), u ≠ v).
//! 3. The unification unify(x,y) is defined iff x,y are compatible, and is computed coordinatewise:
//!      unify(Some(u), Some(u)) = Some(u),
//!      unify(Some(u), None)    = Some(u),
//!      unify(None,    Some(u)) = Some(u),
//!      unify(None,    None)    = None.
//!
//! Synthesis task (formal):
//! 1. Input:
//!    - K ≥ 1 components, each with a finite local-graph structure captured as a sparse
//!      adjacency (only non-masked transitions are stored). For each component i and each
//!      local state s, we have a sparse map s -> { a ↦ s' }. For symbols not in the map,
//!      the successor is masked (⊥).
//!    - An alphabet size M.
//!    - A start point p0 ∈ (S_0 ∪ {⊥}) × ... × (S_{K-1} ∪ {⊥}).
//!
//! 2. Goal:
//!    Produce a finite set R ⊆ (S_0 ∪ {⊥}) × ... × (S_{K-1} ∪ {⊥}) of "representatives"
//!    and a map φ defined on at least:
//!      Dom ⊇ { p0 } ∪ { succ(r, a) | r ∈ R, a ∈ Σ } ∪ { succ(succ(r, a), b) | ... } ...,
//!    such that:
//!      (C1) Well-formedness: ∀x ∈ Dom, x ≤ φ(x).
//!      (C2) Closure/Stability: ∀r ∈ R, ∀a ∈ Σ, succ(r, a) ∈ Dom.
//!      (C3) Start: p0 ∈ Dom.
//!
//! If these hold, we obtain a total transition function on R:
//!   δ: R × Σ → R,   δ(r, a) = φ(succ(r, a)).
//! Different valid (R, φ) choices are possible; smaller |R| is better.
//!
//! Quality objective:
//!   Minimize |R| subject to (C1–C3).
//!
//! Algorithm provided here (optimized greedy):
//! 1. Maintain representatives R and a map φ from product points to indices in R.
//! 2. Initialize with r_0 := p0 and φ(p0) = 0.
//! 3. Perform a worklist exploration. Whenever a successor x arises, if it is not in φ,
//!    find the best existing representative r_j to merge with by minimizing the lexicographic
//!    cost (specificity_increase, current_specificity, j). If none is compatible, create a new
//!    representative r_new := x. If a representative is strengthened, re-enqueue it.
//!
//! Correctness and performance are discussed in the main README and comments below.

#![allow(dead_code)]

use std::collections::{BTreeMap, HashMap, VecDeque, HashSet};

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
                    return None; // Conflict
                }
            }
            (Some(x), None) => out.push(Some(x)),
            (None, Some(y)) => out.push(Some(y)),
            (None, None) => out.push(None),
        }
    }
    Some(out)
}

/// Checks if tuple `x` is less specific than or equal to tuple `y` (x ≤ y).
/// Intuitively, `y` is at least as "specific" as `x`: whenever `x` specifies a concrete local
/// state, `y` agrees. Otherwise `x` leaves the coordinate unspecified (⊥).
fn is_less_or_equal(x: &ProductTuple, y: &ProductTuple) -> bool {
    debug_assert_eq!(x.len(), y.len(), "Tuples must have same arity for comparison");
    for i in 0..x.len() {
        match (x[i], y[i]) {
            (Some(xs), Some(ys)) => {
                if xs != ys {
                    return false; // Conflict: x specifies something different from y
                }
            }
            (Some(_), None) => {
                return false; // x specifies a value, but y is masked (x is more specific than y)
            }
            _ => {
                // (None, Some(_)) or (None, None) are fine
            }
        }
    }
    true
}

/// Compute the successor of a product point under a given symbol, using sparse components.
/// Rule:
/// 1. If the coordinate is None (masked), it stays None.
/// 2. If the coordinate is Some(s), we follow components[i][s][symbol] if present, else return None.
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
    /// (C2) For all r in reps and all symbols a: succ(r, a) ∈ image.
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
            if !is_less_or_equal(x, &self.reps[id]) {
                return Err("image well-formedness violated: x is more specific than its rep".into());
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

/// Optimized greedy synthesizer:
/// Strategy for assigning a new successor x:
/// 1) If x already has a home in image, continue.
/// 2) Otherwise, among all existing reps r_j compatible with x (unify_tuples succeeds),
///    choose the one minimizing the lexicographic cost:
///       cost(j) = (specificity_increase, current_specificity, j)
///    where specificity is the count of Some(_) coordinates.
///    If specificity_increase == 0 (i.e., x ≤ r_j), we accept immediately (best possible).
/// 3) If no compatible rep exists, create a new representative r_new := x.
/// 4) If a rep r_j is strengthened (becomes more specific), re-enqueue it so we maintain closure.
///
/// Correctness (sketch):
/// 1) C3: start is inserted at initialization.
/// 2) C1: Every mapping x → j either has x ≤ reps[j] already, or reps[j] is replaced by unify(reps[j], x),
///    and by definition x ≤ unify(reps[j], x) holds. Existing mappings remain valid because reps[j] only
///    becomes more specific (old reps[j] ≤ new reps[j]).
/// 3) C2: Worklist ensures that after each change to a representative, all of its successors are processed,
///    so for every rep and every symbol, succ(rep, a) ∈ Dom.
pub fn synthesize_greedy(inst: &Instance) -> Solution {
    let _ = inst.validate_shape(); // If invalid, callers can validate first.

    let mut reps: Vec<ProductTuple> = Vec::new();
    let mut image: HashMap<ProductTuple, usize> = HashMap::new();
    let mut work_queue: VecDeque<usize> = VecDeque::new();
    let mut work_set: HashSet<usize> = HashSet::new(); // To track items already in the queue

    // Initialize
    reps.push(inst.start.clone());
    image.insert(inst.start.clone(), 0);
    work_queue.push_back(0);
    work_set.insert(0);

    // Explore representatives
    while let Some(rid) = work_queue.pop_front() {
        work_set.remove(&rid);

        // Clone to avoid borrow issues when mutating reps while iterating
        let rep = reps[rid].clone();

        for a in 0..inst.alphabet_size {
            let x = successor_tuple(&rep, a, &inst.components);

            if image.contains_key(&x) {
                continue;
            }

            // Find the best existing representative to merge with
            let mut best_candidate_id: Option<usize> = None;
            let mut best_cost: (usize, usize, usize) = (usize::MAX, usize::MAX, usize::MAX);

            for j in 0..reps.len() {
                if let Some(unified) = unify_tuples(&reps[j], &x) {
                    let current_spec = reps[j].iter().filter(|v| v.is_some()).count();
                    let unified_spec = unified.iter().filter(|v| v.is_some()).count();
                    let spec_increase = unified_spec - current_spec;

                    let cost = (spec_increase, current_spec, j);

                    // Early acceptance for zero increase (x ≤ reps[j])
                    if spec_increase == 0 {
                        best_candidate_id = Some(j);
                        best_cost = cost;
                        break;
                    }

                    if cost < best_cost {
                        best_cost = cost;
                        best_candidate_id = Some(j);
                    }
                }
            }

            if let Some(j) = best_candidate_id {
                let old_rep = reps[j].clone();
                let unified = unify_tuples(&old_rep, &x).unwrap(); // safe: we filtered with unify_tuples above

                if unified != old_rep {
                    reps[j] = unified;
                    // Re-enqueue j only if not already queued
                    if work_set.insert(j) {
                        work_queue.push_back(j);
                    }
                }
                image.insert(x, j);
                continue;
            }

            // No compatible representative found => create a new one
            let new_id = reps.len();
            reps.push(x.clone());
            image.insert(x, new_id);
            work_queue.push_back(new_id);
            work_set.insert(new_id);
        }
    }

    Solution { reps, image }
}

/// Backwards-compatible wrapper: constructs a solution by the greedy method and
/// returns its representatives and point-map.
///
/// start_tuple: the start point p0
/// components: sparse transitions per component (symbols missing => masked)
/// alphabet_size: |Σ|
///
/// Returns:
/// (reps, point_map) where:
///   reps      = representatives R
///   point_map = φ mapping each visited product successor (and the start) to a representative id
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
    let sol = synthesize_greedy(&inst);
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
    fn test_is_less_or_equal() {
        // x ≤ y
        assert!(is_less_or_equal(&vec![None, None], &vec![Some(1), Some(2)]));
        assert!(is_less_or_equal(&vec![Some(1), None], &vec![Some(1), Some(2)]));
        assert!(is_less_or_equal(&vec![None, Some(2)], &vec![Some(1), Some(2)]));
        assert!(is_less_or_equal(&vec![Some(1), Some(2)], &vec![Some(1), Some(2)]));

        // x not ≤ y
        assert!(!is_less_or_equal(&vec![Some(1), Some(2)], &vec![Some(1), None])) ;   // x is more specific
        assert!(!is_less_or_equal(&vec![Some(1), Some(2)], &vec![Some(1), Some(3)])); // conflict
        assert!(!is_less_or_equal(&vec![Some(1), Some(2)], &vec![None, Some(2)]));    // x is more specific
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
        // Components react to distinct symbols (0 vs 1).
        let comp0 = vec![BTreeMap::from([(0usize, 0usize)])];
        let comp1 = vec![BTreeMap::from([(1usize, 0usize)])];
        let components = vec![comp0, comp1];
        let alphabet_size = 3;
        let start_tuple = vec![Some(0), Some(0)];

        let (states, point_map) =
            merge_and_build_automaton(start_tuple, &components, alphabet_size);

        // The optimized algorithm should still find a single representative in this simple case.
        assert_eq!(states.len(), 1);
        assert_eq!(states[0], vec![Some(0), Some(0)]);

        let succ0 = successor_tuple(&states[0], 0, &components);
        let succ1 = successor_tuple(&states[0], 1, &components);
        let succ2 = successor_tuple(&states[0], 2, &components);
        assert_eq!(*point_map.get(&succ0).unwrap(), 0);
        assert_eq!(*point_map.get(&succ1).unwrap(), 0);
        assert_eq!(*point_map.get(&succ2).unwrap(), 0);
    }

    #[test]
    fn test_verify_solution_constraints() {
        // Components: both have 2 local states: 0 and 1.
        // For comp0: symbol 0 toggles 0->1, 1->1; symbol 1 no-op (masked).
        // For comp1: symbol 1 toggles 0->1, 1->1; symbol 0 no-op (masked).
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

        let sol = synthesize_greedy(&inst);
        sol.verify(&inst).expect("greedy solution must satisfy constraints");

        // Minimality check on this specific topology:
        // Start (0,0). The successors force at least two representatives: (0,0) and (1,1).
        assert!(sol.reps.len() >= 2);
    }
}
