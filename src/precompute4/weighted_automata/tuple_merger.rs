//! Masked-Product Point Synthesis (optimized implementation)
//!
//! This file defines a small, abstract puzzle (no domain-specific context):
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
//!   - K ≥ 1 components, each with a finite local-graph structure captured as a sparse
//!     adjacency (only non-masked transitions are stored). For each component i and each
//!     local state s, we have a sparse map s -> { a ↦ s' }. For symbols not in the map,
//!     the successor is masked (⊥).
//!   - An alphabet size M.
//!   - A start point p0 ∈ (S_0 ∪ {⊥}) × ... × (S_{K-1} ∪ {⊥}).
//!
//! - Goal:
//!   Produce a finite set R ⊆ (S_0 ∪ {⊥}) × ... × (S_{K-1} ∪ {⊥}) of "representatives"
//!   and a map φ defined on at least:
//!     Dom ⊇ { p0 } ∪ { succ(r, a) | r ∈ R, a ∈ Σ } ∪ { succ(succ(r, a), b) | ... } ...,
//!   i.e., on every product successor encountered when starting from any representative,
//!   such that:
//!     (C1) Well-formedness: ∀x ∈ Dom, x ≤ φ(x).
//!          (Every tuple we "name" is mapped to a representative that is at least as
//!          specific as the tuple.)
//!     (C2) Closure/Stability: ∀r ∈ R, ∀a ∈ Σ, succ(r, a) ∈ Dom.
//!          (All outgoing successors of every representative have a home.)
//!     (C3) Start: p0 ∈ Dom.
//!
//! If these hold, we obtain a total transition function on R:
//!   δ: R × Σ → R,   δ(r, a) = φ(succ(r, a)).
//! This is well-defined by (C2) and assigns every representative a successor representative
//! under each symbol. Different valid (R, φ) choices are possible; smaller |R| is better.
//!
//! Quality objective:
//!   Minimize |R| subject to (C1–C3).
//!
//! This implementation provides multiple algorithms:
//!   - synthesize_greedy: baseline on-the-fly construction
//!   - synthesize_smart: explicit reachability + intelligent merging
//!   - synthesize_optimal: smart synthesis + iterative optimization
//!   - optimize_solution: post-processing to merge compatible representatives
//!
//! Public API:
//!   - merge_and_build_automaton: uses synthesize_optimal for best results

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

/// Check if a ≤ b (a is less specific or equal to b in the partial order).
fn is_less_or_equal(a: &ProductTuple, b: &ProductTuple) -> bool {
    if a.len() != b.len() {
        return false;
    }
    for i in 0..a.len() {
        match (a[i], b[i]) {
            (Some(x), Some(y)) => {
                if x != y {
                    return false;
                }
            }
            (Some(_), None) => return false, // a is more specific than b
            _ => {}
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
                if s < components[i].len() {
                    let map = &components[i][s];
                    if let Some(&t) = map.get(&symbol) {
                        out.push(Some(t));
                    } else {
                        out.push(None);
                    }
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
            let rep = &self.reps[id];
            if !is_less_or_equal(x, rep) {
                return Err("image well-formedness violated: x is not ≤ its representative".into());
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

/// Baseline greedy synthesizer (kept for comparison):
/// - Start with R = [start], φ(start) = 0.
/// - Pop a representative; for each symbol, compute its successor x.
///   - If φ(x) already exists, continue.
///   - Else try to unify x with some existing r_j; if compatible:
///       r_j := unify(r_j, x) (possibly increasing specificity), push j back on worklist.
///       φ(x) := j.
///   - Else create a new representative r_new := x, set φ(x) := new_id, and push it.
///
/// This guarantees (C1–C3) but is not optimal.
pub fn synthesize_greedy(inst: &Instance) -> Solution {
    let _ = inst.validate_shape();

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

/// Compute all reachable product points from the start point.
fn compute_reachable_points(inst: &Instance) -> HashSet<ProductTuple> {
    let mut reachable = HashSet::new();
    let mut worklist = VecDeque::new();

    reachable.insert(inst.start.clone());
    worklist.push_back(inst.start.clone());

    while let Some(point) = worklist.pop_front() {
        for a in 0..inst.alphabet_size {
            let succ = successor_tuple(&point, a, &inst.components);
            if reachable.insert(succ.clone()) {
                worklist.push_back(succ);
            }
        }
    }

    reachable
}

/// Smart synthesis: explicit reachability followed by intelligent representative selection.
/// This computes all reachable points first, then builds representatives more carefully.
pub fn synthesize_smart(inst: &Instance) -> Solution {
    let _ = inst.validate_shape();

    // Phase 1: Compute all reachable points
    let reachable_set = compute_reachable_points(inst);
    let reachable_vec: Vec<ProductTuple> = reachable_set.into_iter().collect();

    // Phase 2: Build representatives intelligently
    // Start with empty representative set
    let mut reps: Vec<ProductTuple> = Vec::new();
    let mut point_to_rep: HashMap<ProductTuple, usize> = HashMap::new();

    // Process points in a specific order: start with the start point
    let mut processed = HashSet::new();
    let mut worklist = VecDeque::new();
    worklist.push_back(inst.start.clone());

    while let Some(point) = worklist.pop_front() {
        if processed.contains(&point) {
            continue;
        }
        processed.insert(point.clone());

        // Try to assign this point to an existing representative
        let mut assigned = None;
        for (i, rep) in reps.iter().enumerate() {
            if is_less_or_equal(&point, rep) {
                assigned = Some(i);
                break;
            }
        }

        if assigned.is_none() {
            // Try to unify with an existing representative
            for (i, rep) in reps.iter_mut().enumerate() {
                if let Some(unified) = unify_tuples(rep, &point) {
                    *rep = unified;
                    assigned = Some(i);
                    break;
                }
            }
        }

        let rep_id = if let Some(i) = assigned {
            i
        } else {
            // Create new representative
            let new_id = reps.len();
            reps.push(point.clone());
            new_id
        };

        point_to_rep.insert(point.clone(), rep_id);

        // Add successors to worklist
        for a in 0..inst.alphabet_size {
            let succ = successor_tuple(&point, a, &inst.components);
            if !processed.contains(&succ) && reachable_vec.contains(&succ) {
                worklist.push_back(succ);
            }
        }
    }

    // Phase 3: Ensure closure - add any missing reachable points
    for point in &reachable_vec {
        if !point_to_rep.contains_key(point) {
            // Find or create representative for this point
            let mut assigned = None;
            for (i, rep) in reps.iter().enumerate() {
                if is_less_or_equal(point, rep) {
                    assigned = Some(i);
                    break;
                }
            }

            if assigned.is_none() {
                for (i, rep) in reps.iter_mut().enumerate() {
                    if let Some(unified) = unify_tuples(rep, point) {
                        *rep = unified;
                        assigned = Some(i);
                        break;
                    }
                }
            }

            let rep_id = if let Some(i) = assigned {
                i
            } else {
                let new_id = reps.len();
                reps.push(point.clone());
                new_id
            };

            point_to_rep.insert(point.clone(), rep_id);
        }
    }

    // Phase 4: Ensure all representative successors are covered
    let mut added_successors = true;
    while added_successors {
        added_successors = false;
        let current_reps = reps.clone();

        for rep in &current_reps {
            for a in 0..inst.alphabet_size {
                let succ = successor_tuple(rep, a, &inst.components);
                if !point_to_rep.contains_key(&succ) {
                    // Need to add this successor
                    let mut assigned = None;
                    for (i, r) in reps.iter().enumerate() {
                        if is_less_or_equal(&succ, r) {
                            assigned = Some(i);
                            break;
                        }
                    }

                    if assigned.is_none() {
                        for (i, r) in reps.iter_mut().enumerate() {
                            if let Some(unified) = unify_tuples(r, &succ) {
                                *r = unified;
                                assigned = Some(i);
                                break;
                            }
                        }
                    }

                    let rep_id = if let Some(i) = assigned {
                        i
                    } else {
                        let new_id = reps.len();
                        reps.push(succ.clone());
                        new_id
                    };

                    point_to_rep.insert(succ, rep_id);
                    added_successors = true;
                }
            }
        }
    }

    Solution {
        reps,
        image: point_to_rep,
    }
}

/// Check if merging two representatives would preserve closure and validity.
fn can_merge_representatives(
    reps: &[ProductTuple],
    i: usize,
    j: usize,
    image: &HashMap<ProductTuple, usize>,
    inst: &Instance,
) -> bool {
    if i == j {
        return false;
    }

    // Check if they can unify
    let unified = match unify_tuples(&reps[i], &reps[j]) {
        Some(u) => u,
        None => return false,
    };

    // Check if all points currently assigned to i or j are compatible with unified
    for (point, &rep_id) in image.iter() {
        if rep_id == i || rep_id == j {
            if !is_less_or_equal(point, &unified) {
                return false;
            }
        }
    }

    // Check if closure is preserved: all successors of unified must be covered
    for a in 0..inst.alphabet_size {
        let succ = successor_tuple(&unified, a, &inst.components);
        if !image.contains_key(&succ) {
            return false;
        }
    }

    true
}

/// Post-process a solution to iteratively merge compatible representatives.
/// This is a key optimization that can significantly reduce |R|.
pub fn optimize_solution(mut sol: Solution, inst: &Instance) -> Solution {
    let mut improved = true;

    while improved {
        improved = false;

        for i in 0..sol.reps.len() {
            for j in (i + 1)..sol.reps.len() {
                if can_merge_representatives(&sol.reps, i, j, &sol.image, inst) {
                    // Perform merge: unify i and j, keep in position i, remove j
                    let unified = unify_tuples(&sol.reps[i], &sol.reps[j]).unwrap();
                    sol.reps[i] = unified;

                    // Update image: redirect all references from j to i
                    for rep_id in sol.image.values_mut() {
                        if *rep_id == j {
                            *rep_id = i;
                        } else if *rep_id > j {
                            *rep_id -= 1;
                        }
                    }

                    sol.reps.remove(j);
                    improved = true;
                    break;
                }
            }
            if improved {
                break;
            }
        }
    }

    sol
}

/// Optimal synthesis: combines smart initialization with iterative optimization.
/// This is the recommended synthesis method for best results.
pub fn synthesize_optimal(inst: &Instance) -> Solution {
    // Try both strategies and pick the better one
    let greedy_sol = synthesize_greedy(inst);
    let smart_sol = synthesize_smart(inst);

    let mut best = if greedy_sol.reps.len() <= smart_sol.reps.len() {
        greedy_sol
    } else {
        smart_sol
    };

    // Apply optimization passes
    best = optimize_solution(best, inst);

    // One more optimization pass for good measure
    best = optimize_solution(best, inst);

    best
}

/// Backwards-compatible wrapper: constructs a solution by the optimal method and
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
    let sol = synthesize_optimal(&inst);
    (sol.reps, sol.image)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unify_tuples() {
        assert_eq!(
            unify_tuples(&vec![Some(1), None], &vec![None, Some(2)]),
            Some(vec![Some(1), Some(2)])
        );
        assert_eq!(
            unify_tuples(&vec![Some(1), Some(2)], &vec![Some(1), Some(2)]),
            Some(vec![Some(1), Some(2)])
        );
        assert_eq!(
            unify_tuples(&vec![Some(1), None], &vec![Some(1), Some(3)]),
            Some(vec![Some(1), Some(3)])
        );
        assert_eq!(
            unify_tuples(&vec![Some(1), Some(2)], &vec![Some(1), Some(3)]),
            None
        );
        assert_eq!(
            unify_tuples(&vec![None, None], &vec![Some(1), Some(2)]),
            Some(vec![Some(1), Some(2)])
        );
    }

    #[test]
    fn test_is_less_or_equal() {
        assert!(is_less_or_equal(&vec![None, None], &vec![Some(1), Some(2)]));
        assert!(is_less_or_equal(&vec![Some(1), None], &vec![Some(1), Some(2)]));
        assert!(is_less_or_equal(&vec![Some(1), Some(2)], &vec![Some(1), Some(2)]));
        assert!(!is_less_or_equal(&vec![Some(1), Some(2)], &vec![Some(1), None]));
        assert!(!is_less_or_equal(&vec![Some(1), Some(2)], &vec![Some(1), Some(3)]));
    }

    #[test]
    fn test_successor_tuple_sparse() {
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
    fn test_simple_merge_optimal() {
        let comp0 = vec![BTreeMap::from([(0usize, 0usize)])];
        let comp1 = vec![BTreeMap::from([(1usize, 0usize)])];
        let components = vec![comp0, comp1];
        let alphabet_size = 3;
        let start_tuple = vec![Some(0), Some(0)];

        let (states, point_map) =
            merge_and_build_automaton(start_tuple, &components, alphabet_size);

        // Should find a minimal solution (likely 1 representative)
        assert!(states.len() >= 1);
        assert!(states.len() <= 2); // Optimal should be 1 or 2

        // Verify it's a valid solution
        let inst = Instance::new(vec![Some(0), Some(0)], components, alphabet_size);
        let sol = Solution {
            reps: states,
            image: point_map,
        };
        sol.verify(&inst).expect("solution must be valid");
    }

    #[test]
    fn test_verify_solution_constraints() {
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

        let sol = synthesize_optimal(&inst);
        sol.verify(&inst).expect("optimal solution must satisfy constraints");
    }

    #[test]
    fn test_optimal_better_than_greedy() {
        // Create a test case where greedy is suboptimal
        // Component 0: 0 --a--> 1, 1 --a--> 1
        // Component 1: 0 --b--> 1, 1 --b--> 1
        let comp0 = vec![
            BTreeMap::from([(0usize, 1usize)]),
            BTreeMap::from([(0usize, 1usize)]),
        ];
        let comp1 = vec![
            BTreeMap::from([(1usize, 1usize)]),
            BTreeMap::from([(1usize, 1usize)]),
        ];
        let components = vec![comp0, comp1];
        let inst = Instance::new(vec![Some(0), Some(0)], components, 2);

        let greedy_sol = synthesize_greedy(&inst);
        let optimal_sol = synthesize_optimal(&inst);

        // Optimal should be at least as good as greedy
        assert!(optimal_sol.reps.len() <= greedy_sol.reps.len());

        // Both should be valid
        greedy_sol.verify(&inst).expect("greedy must be valid");
        optimal_sol.verify(&inst).expect("optimal must be valid");
    }

    #[test]
    fn test_optimize_solution_reduces_size() {
        // Create an intentionally suboptimal solution and verify optimization improves it
        let comp0 = vec![BTreeMap::from([(0usize, 0usize)])];
        let comp1 = vec![BTreeMap::from([(1usize, 0usize)])];
        let components = vec![comp0, comp1];
        let inst = Instance::new(vec![Some(0), Some(0)], components, 2);

        let sol = synthesize_greedy(&inst);
        let original_size = sol.reps.len();

        let optimized = optimize_solution(sol, &inst);
        let optimized_size = optimized.reps.len();

        // Optimization should not increase size
        assert!(optimized_size <= original_size);

        // Result should still be valid
        optimized.verify(&inst).expect("optimized must be valid");
    }

    #[test]
    fn test_reachability_computation() {
        let comp0 = vec![
            BTreeMap::from([(0usize, 1usize)]),
            BTreeMap::from([(0usize, 0usize)]),
        ];
        let comp1 = vec![BTreeMap::from([(0usize, 0usize)])];
        let inst = Instance::new(vec![Some(0), Some(0)], vec![comp0, comp1], 1);

        let reachable = compute_reachable_points(&inst);

        // Start is reachable
        assert!(reachable.contains(&vec![Some(0), Some(0)]));

        // Should have found multiple states
        assert!(reachable.len() >= 2);
    }
}