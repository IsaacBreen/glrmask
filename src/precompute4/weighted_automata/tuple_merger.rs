//! Masked-Product Point Synthesis (self-contained spec and optimized greedy)
//!
//! Problem (informal):
//! - There are K components. Each component has a finite set of local states S_i plus a
//!   special "masked" value ⊥ (read: off/irrelevant).
//! - We have an alphabet of symbols Σ = {0,1,...,M-1}.
//! - For each component i and each local state s ∈ S_i, we define a partial transition
//!   τ_i(s, a) ∈ S_i on symbol a. If a transition does not exist, we treat it as masked (⊥).
//!   When a coordinate is already masked (⊥), it remains masked under any symbol.
//!
//! A product point is a K-tuple p = (p_0, ..., p_{K-1}) where each coordinate p_i ∈ S_i ∪ {⊥}.
//! A product successor is defined componentwise: succ(p, a)_i =
//!   - τ_i(p_i, a) if p_i ∈ S_i and the symbol a-transition exists;
//!   - ⊥ otherwise (including when p_i = ⊥).
//!
//! We write ⊥ as None in code; some local state x ∈ S_i is encoded as Some(x).
//!
//! Compatibility and Unification:
//! - Define a quasiorder ≤ on points: x ≤ y iff for all i, either x_i = ⊥ or x_i = y_i.
//!   Intuitively, y is at least as "specific" as x: whenever x specifies a concrete local
//!   state, y agrees. Otherwise x leaves the coordinate unspecified (⊥).
//! - Two points x,y are compatible iff they do not conflict on specified coordinates,
//!   i.e., ∀i: not (x_i = Some(u), y_i = Some(v), u ≠ v).
//! - The unification (least upper bound) unify(x,y) is defined iff x,y are compatible,
//!   and is computed coordinatewise:
//!     unify(Some(u), Some(u)) = Some(u),
//!     unify(Some(u), None)    = Some(u),
//!     unify(None,    Some(u)) = Some(u),
//!     unify(None,    None)    = None.
//!
//! Synthesis task (formal):
//! - Input:
//!   - K ≥ 1 components, sparse encoding (symbols missing ⇒ masked).
//!   - Alphabet size M.
//!   - Start point p0.
//! - Output:
//!   - Representatives R and a map φ defined on at least Start ∪ Succ-closure(R) such that:
//!     (C1) ∀x ∈ Dom, x ≤ φ(x).
//!     (C2) ∀r ∈ R, ∀a ∈ Σ, succ(r, a) ∈ Dom.
//!     (C3) p0 ∈ Dom.
//!
//! Quality objective (primary):
//!   Minimize |R|, subject to (C1–C3).
//!
//! This implementation provides an optimized greedy synthesizer designed from the following
//! hypotheses and proofs:
//!   - H1 (quality heuristic): When mapping a new successor x to an existing representative r_j,
//!     choosing the candidate that minimizes the number of None→Some flips (specificity increase)
//!     tends to keep r_j general and increases the chance that r_j covers future successors,
//!     reducing |R| in practice.
//!   - H2 (tie-break): When multiple r_j induce the same minimal specificity increase, prefer the
//!     r_j with the smallest current specificity (fewest Some). This further preserves generality.
//!
//! Correctness arguments (sketch):
//!   - (C1) If x is assigned to r_j with zero increase, then x ≤ r_j. If a specialization is
//!     required, r_j is updated to unify(r_j, x); by definition, x ≤ unify(r_j, x). Thus x ≤ φ(x).
//!   - (C2) Representatives are processed via a worklist. Whenever r_j is specialized, we re-enqueue
//!     it; the loop continues until all succ(r, a) have entries in φ, guaranteeing closure.
//!   - (C3) Initialization inserts p0 into image, hence p0 ∈ Dom.
//!   - Termination: The product space is finite. Each specialization increases specificity and cannot
//!     be undone; finite total increases exist. When no more specializations or creations are needed,
//!     the worklist empties and the algorithm stops.
//!
//! Public API (unchanged):
//!   - type ProductTuple
//!   - fn successor_tuple(...)
//!   - fn merge_and_build_automaton(...) -> (Vec<ProductTuple>, HashMap<ProductTuple, usize>)
//!     (wraps the optimized greedy synthesizer).

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
            (None, Some(_)) => {
                // x is masked while y is specified: ok (y is ≥ x)
            }
            (None, None) => {
                // both masked: ok
            }
        }
    }
    true
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

/// Optimized greedy synthesizer that prioritizes minimizing |R| while remaining fast.
///
/// Strategy per successor x:
///   - Evaluate all existing representatives r_j that are compatible with x (unification succeeds).
///   - Choose the one that minimizes (specificity_increase, current_specificity), where:
///       specificity_increase = (#Some in unify(r_j, x)) - (#Some in r_j)
///       current_specificity  = #Some in r_j
///     This preserves generality and tends to reduce future representative creations.
///   - If no compatible representative exists, create a new one.
///   - Re-enqueue a representative if it was specialized (so all its succ are reprocessed).
pub fn synthesize_greedy(inst: &Instance) -> Solution {
    let _ = inst.validate_shape(); // Callers can validate first; we proceed regardless.

    let mut reps: Vec<ProductTuple> = Vec::new();
    let mut image: HashMap<ProductTuple, usize> = HashMap::new();
    let mut work_queue: VecDeque<usize> = VecDeque::new();
    let mut work_set: HashSet<usize> = HashSet::new(); // Track items currently queued

    // Initialize with start
    reps.push(inst.start.clone());
    image.insert(inst.start.clone(), 0);
    work_queue.push_back(0);
    work_set.insert(0);

    // Explore until fixpoint
    while let Some(rid) = work_queue.pop_front() {
        work_set.remove(&rid);

        let rep_snapshot = reps[rid].clone(); // Avoid mutable aliasing while scanning
        for a in 0..inst.alphabet_size {
            let x = successor_tuple(&rep_snapshot, a, &inst.components);
            if image.contains_key(&x) {
                continue; // Already mapped
            }

            // Find best existing representative to absorb x
            // Store (rep_id, unified_tuple, (spec_increase, current_spec))
            let mut best: Option<(usize, ProductTuple, (usize, usize))> = None;

            for j in 0..reps.len() {
                if let Some(unified) = unify_tuples(&reps[j], &x) {
                    let current_spec = reps[j].iter().filter(|v| v.is_some()).count();
                    let unified_spec = unified.iter().filter(|v| v.is_some()).count();
                    let inc = unified_spec - current_spec;

                    let cost = (inc, current_spec);
                    match best {
                        Some((_, _, best_cost)) => {
                            if cost < best_cost {
                                best = Some((j, unified, cost));
                            }
                        }
                        None => {
                            best = Some((j, unified, cost));
                        }
                    }
                }
            }

            if let Some((j, unified_tuple, (inc, _cur_spec))) = best {
                // Absorb x into representative j
                if inc > 0 || unified_tuple != reps[j] {
                    // Specialization occurred; update and re-enqueue j if not already queued
                    reps[j] = unified_tuple;
                    if work_set.insert(j) {
                        work_queue.push_back(j);
                    }
                }
                image.insert(x, j);
            } else {
                // No compatible representative: create new one
                let new_id = reps.len();
                reps.push(x.clone());
                image.insert(x, new_id);
                if work_set.insert(new_id) {
                    work_queue.push_back(new_id);
                }
            }
        }
    }

    Solution { reps, image }
}

/// Backwards-compatible wrapper: constructs a solution by the optimized greedy method and
/// returns its representatives and point-map.
///
/// - start_tuple: the start point p0
/// - components: sparse transitions per component (symbols missing => masked)
/// - alphabet_size: |Σ|
///
/// Returns:
/// - (reps, point_map) where:
///     reps      = representatives R
///     point_map = φ mapping each visited product successor (and the start) to a representative id
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
        // x ≰ y
        assert!(!is_less_or_equal(&vec![Some(1), Some(2)], &vec![Some(1), None]));
        assert!(!is_less_or_equal(&vec![Some(1), Some(2)], &vec![Some(1), Some(3)]));
        assert!(!is_less_or_equal(&vec![Some(1), Some(2)], &vec![None, Some(2)]));
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

        // The optimized algorithm should find a single representative here.
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

        let sol = synthesize_greedy(&inst);
        sol.verify(&inst).expect("greedy solution must satisfy constraints");

        // Trace for this scenario (intuitive):
        // Start: (0,0)
        // succ(0): (1,⊥)  -> cannot be absorbed by (0,0), create new r1
        // succ(1): (⊥,1)  -> unifies with r1 to (1,1), reducing |R| to 2
        assert_eq!(sol.reps.len(), 2);
        assert_eq!(sol.reps[0], vec![Some(0), Some(0)]);
        assert_eq!(sol.reps[1], vec![Some(1), Some(1)]);
    }
}
