//! Masked-Product Point Synthesis (self-contained spec and baseline)
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
//! Quality objective (to optimize in future work):
//!   Minimize |R| subject to (C1–C3).
//!
//! Baseline algorithm (provided here):
//!   synthesize_greedy implements a simple, deterministic construction:
//!   - Maintain a growing vector of representatives R = [r_0, r_1, ...] and a map φ from
//!     product points to indices in R.
//!   - Initialize with r_0 := p0 and φ(p0) = 0.
//!   - Perform a worklist exploration of representatives. Whenever a new product successor
//!     x arises, either assign it to an existing representative r_j that can unify with x
//!     (updating r_j := unify(r_j, x) if this increases specificity), or create a fresh
//!     representative r_new := x.
//!   This satisfies (C1–C3), but is not guaranteed to be minimal.
//!
//! Verification:
//!   The struct Solution offers verify(&Instance) -> Result<(), String> to mechanically
//!   check (C1–C3). Any alternative algorithm can output (reps, image) and be verified the
//!   same way.
//!
//! Public API kept for backward compatibility:
//!   - type ProductTuple
//!   - fn successor_tuple(...)
//!   - fn merge_and_build_automaton(...) -> (Vec<ProductTuple>, HashMap<ProductTuple, usize>)
//!     which simply wraps the greedy synthesizer.

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

/// Improved greedy synthesizer:
/// - Prioritizes assigning a successor `x` to an existing representative `r_j` if `x ≤ r_j` (no change to `r_j`).
/// - If no such `r_j` exists, it finds an `r_j` that requires the minimal increase in specificity when unified with `x`.
/// - Only if no compatible `r_j` is found, a new representative `r_new = x` is created.
/// This strategy aims to minimize `|R|` while satisfying (C1–C3).
pub fn synthesize_greedy(inst: &Instance) -> Solution {
    let _ = inst.validate_shape(); // If invalid, we'll likely fail later; callers can validate first.

    let mut reps: Vec<ProductTuple> = Vec::new();
    let mut image: HashMap<ProductTuple, usize> = HashMap::new();
    let mut work_queue: VecDeque<usize> = VecDeque::new();
    let mut work_set: HashSet<usize> = HashSet::new(); // To track items in work_queue efficiently

    // Initialize
    reps.push(inst.start.clone());
    image.insert(inst.start.clone(), 0);
    work_queue.push_back(0);
    work_set.insert(0);

    // Explore representatives
    while let Some(rid) = work_queue.pop_front() {
        work_set.remove(&rid); // Remove from set when processing

        // Clone the representative to avoid mutable borrow issues while iterating reps
        // and potentially modifying reps[j] or adding new reps.
        let rep = reps[rid].clone();

        for a in 0..inst.alphabet_size {
            let x = successor_tuple(&rep, a, &inst.components);

            if image.contains_key(&x) {
                continue;
            }

            let mut best_candidate_id: Option<usize> = None;
            let mut best_specificity_increase: usize = usize::MAX; // Use usize::MAX for "infinity"

            // Iterate through existing representatives to find the best candidate for 'x'
            for j in 0..reps.len() {
                if let Some(unified) = unify_tuples(&reps[j], &x) {
                    let current_specificity = reps[j].iter().filter(|&v| v.is_some()).count();
                    let unified_specificity = unified.iter().filter(|&v| v.is_some()).count();
                    let specificity_increase = unified_specificity - current_specificity;

                    // Prioritize zero increase (x <= reps[j])
                    if specificity_increase == 0 {
                        best_specificity_increase = 0;
                        best_candidate_id = Some(j);
                        break; // Found an ideal candidate, no need to look further
                    }

                    // For non-zero increase, compare and potentially update
                    if specificity_increase < best_specificity_increase {
                        best_specificity_increase = specificity_increase;
                        best_candidate_id = Some(j);
                    } else if specificity_increase == best_specificity_increase {
                        // Tie-breaking: prefer smaller index for determinism
                        if let Some(current_best_j) = best_candidate_id {
                            if j < current_best_j {
                                best_candidate_id = Some(j);
                            }
                        }
                    }
                }
            }

            let home = if let Some(j) = best_candidate_id {
                let old_rep = reps[j].clone();
                let unified = unify_tuples(&old_rep, &x).unwrap(); // Must be Some, as j was a valid candidate

                if unified != old_rep {
                    reps[j] = unified;
                    if work_set.insert(j) { // Add to set and queue if not already present
                        work_queue.push_back(j);
                    }
                }
                j
            } else {
                // No compatible representative found, create a new one.
                let new_id = reps.len();
                reps.push(x.clone());
                work_queue.push_back(new_id);
                work_set.insert(new_id);
                new_id
            };

            image.insert(x, home);
        }
    }

    Solution { reps, image }
}

/// Backwards-compatible wrapper: constructs a solution by the greedy method and
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

        // The greedy algorithm finds a single representative, because both
        // successors unify back into the same representative.
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

        let sol = synthesize_greedy(&inst);
        sol.verify(&inst).expect("greedy solution must satisfy constraints");
    }
}