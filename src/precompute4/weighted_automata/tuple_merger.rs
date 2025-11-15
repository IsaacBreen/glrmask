//! Masked-Product Point Synthesis (self-contained spec and baseline)

#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};

pub type Symbol = usize;
pub type LocalState = usize;
pub type ProductTuple = Vec<Option<LocalState>>;
pub type SparseComponent = Vec<BTreeMap<Symbol, LocalState>>;
pub type Components = Vec<SparseComponent>;

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

#[inline]
fn merge_spec_increase_if_compatible(a: &ProductTuple, b: &ProductTuple) -> Option<usize> {
    if a.len() != b.len() {
        return None;
    }
    let mut inc = 0usize;
    for i in 0..a.len() {
        match (a[i], b[i]) {
            (Some(x), Some(y)) => {
                if x != y {
                    return None;
                }
            }
            (None, Some(_)) => inc += 1,
            _ => {}
        }
    }
    Some(inc)
}

fn is_less_or_equal(a: &ProductTuple, b: &ProductTuple) -> bool {
    debug_assert_eq!(a.len(), b.len(), "Tuples must have same arity for comparison");
    for i in 0..a.len() {
        match (a[i], b[i]) {
            (Some(xs), Some(ys)) => {
                if xs != ys {
                    return false;
                }
            }
            (Some(_), None) => {
                return false;
            }
            _ => {}
        }
    }
    true
}

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

type DenseComponent = Vec<Vec<Option<LocalState>>>;
type DenseComponents = Vec<DenseComponent>;

fn build_dense_components(components: &[SparseComponent], alphabet_size: usize) -> DenseComponents {
    let mut dense: DenseComponents = Vec::with_capacity(components.len());
    for comp in components {
        let mut dc: DenseComponent = Vec::with_capacity(comp.len());
        for state_map in comp {
            let mut row = vec![None; alphabet_size];
            for (&sym, &succ) in state_map {
                if sym < alphabet_size {
                    row[sym] = Some(succ);
                }
            }
            dc.push(row);
        }
        dense.push(dc);
    }
    dense
}

fn successor_tuple_dense(tuple: &ProductTuple, symbol: Symbol, dense: &DenseComponents) -> ProductTuple {
    let k = dense.len();
    let mut out = Vec::with_capacity(k);
    for i in 0..k {
        match tuple.get(i).copied().flatten() {
            Some(s) => {
                if let Some(t) = dense[i][s][symbol] {
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

pub struct Instance {
    pub start: ProductTuple,
    pub components: Components,
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
        for (i, comp) in self.components.iter().enumerate() {
            if comp.is_empty() {
                return Err(format!("component {} has no local states", i));
            }
        }
        Ok(())
    }
}

pub struct Solution {
    pub reps: Vec<ProductTuple>,
    pub image: HashMap<ProductTuple, usize>,
}

impl Solution {
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

        for (x, &id) in &self.image {
            if id >= self.reps.len() {
                return Err("image maps to out-of-range representative index".into());
            }
            if x.len() != k {
                return Err("a tuple in image has wrong arity".into());
            }
            if !is_less_or_equal(x, &self.reps[id]) {
                return Err("image well-formedness violated: x is not <= its representative".into());
            }
        }

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

struct CandidateIndex {
    present: Vec<HashMap<LocalState, Vec<usize>>>,
    masked: Vec<Vec<usize>>,
    spec_buckets: Vec<BTreeSet<usize>>,
}

impl CandidateIndex {
    fn new(k: usize) -> Self {
        Self {
            present: (0..k).map(|_| HashMap::new()).collect(),
            masked: vec![Vec::new(); k],
            spec_buckets: vec![BTreeSet::new(); k + 1],
        }
    }

    fn on_new_rep(&mut self, rid: usize, tuple: &ProductTuple, spec: usize) {
        for (i, &coord) in tuple.iter().enumerate() {
            match coord {
                Some(v) => self.present[i].entry(v).or_default().push(rid),
                None => self.masked[i].push(rid),
            }
        }
        self.spec_buckets[spec].insert(rid);
    }

    fn on_rep_spec_increase(&mut self, rid: usize, old_spec: usize, new_spec: usize) {
        if old_spec != new_spec {
            let _ = self.spec_buckets[old_spec].remove(&rid);
            self.spec_buckets[new_spec].insert(rid);
        }
    }

    fn on_rep_coordinate_becomes_some(&mut self, rid: usize, i: usize, v: LocalState) {
        self.present[i].entry(v).or_default().push(rid);
    }

    fn argmin_rep_for_all_none(&self) -> Option<usize> {
        for bucket in &self.spec_buckets {
            if let Some(&rid) = bucket.iter().next() {
                return Some(rid);
            }
        }
        None
    }
}

pub fn synthesize_greedy(inst: &Instance) -> Solution {
    let _ = inst.validate_shape();

    const VERBOSE: bool = false;

    let k = inst.components.len();
    let dense_components = build_dense_components(&inst.components, inst.alphabet_size);

    let mut reps: Vec<ProductTuple> = Vec::with_capacity(256);
    let mut image: HashMap<ProductTuple, usize> = HashMap::with_capacity(1024);
    let mut work_queue: VecDeque<usize> = VecDeque::with_capacity(256);
    let mut work_set: HashSet<usize> = HashSet::with_capacity(256);
    let mut rep_spec: Vec<usize> = Vec::with_capacity(256);
    let mut index = CandidateIndex::new(k);

    reps.push(inst.start.clone());
    let start_spec = inst.start.iter().filter(|opt| opt.is_some()).count();
    rep_spec.push(start_spec);
    image.insert(inst.start.clone(), 0);
    work_queue.push_back(0);
    work_set.insert(0);
    index.on_new_rep(0, &inst.start, start_spec);
    if VERBOSE {
        println!("Starting greedy synthesis...");
    }

    while let Some(rid) = work_queue.pop_front() {
        work_set.remove(&rid);

        for a in 0..inst.alphabet_size {
            let x = successor_tuple_dense(&reps[rid], a, &dense_components);
            if image.contains_key(&x) {
                continue;
            }

            let mut best_j: Option<usize> = None;
            let mut best_cost: (usize, usize) = (usize::MAX, usize::MAX);

            let mut coords: Vec<(usize, LocalState)> = Vec::new();
            for i in 0..k {
                if let Some(v) = x[i] {
                    coords.push((i, v));
                }
            }

            let mut best_spec_increase = 0usize;

            if coords.is_empty() {
                if let Some(j) = index.argmin_rep_for_all_none() {
                    best_j = Some(j);
                    best_cost = (0, rep_spec[j]);
                    best_spec_increase = 0;
                }
            } else {
                let mut anchor_idx = 0usize;
                let mut anchor_size = {
                    let (ai, av) = coords[0];
                    let present_len = index.present[ai].get(&av).map(|v| v.len()).unwrap_or(0);
                    present_len + index.masked[ai].len()
                };
                for (idx, &(ai, av)) in coords.iter().enumerate().skip(1) {
                    let present_len = index.present[ai].get(&av).map(|v| v.len()).unwrap_or(0);
                    let size = present_len + index.masked[ai].len();
                    if size < anchor_size {
                        anchor_size = size;
                        anchor_idx = idx;
                    }
                }
                let (ai, av) = coords[anchor_idx];

                let mut consider = |j: usize,
                                    best_j: &mut Option<usize>,
                                    best_cost: &mut (usize, usize),
                                    best_spec_increase: &mut usize| {
                    if let Some(spec_increase) = merge_spec_increase_if_compatible(&reps[j], &x) {
                        let current_cost = (spec_increase, rep_spec[j]);
                        if current_cost < *best_cost || (current_cost == *best_cost && Some(j) < *best_j) {
                            *best_cost = current_cost;
                            *best_j = Some(j);
                            *best_spec_increase = spec_increase;
                        }
                    }
                };

                if let Some(list) = index.present[ai].get(&av) {
                    for &j in list {
                        if reps[j][ai] == Some(av) {
                            consider(j, &mut best_j, &mut best_cost, &mut best_spec_increase);
                        }
                    }
                }
                for &j in &index.masked[ai] {
                    if reps[j][ai].is_none() {
                        consider(j, &mut best_j, &mut best_cost, &mut best_spec_increase);
                    }
                }
            }

            if let Some(j) = best_j {
                let unified_tuple = unify_tuples(&reps[j], &x).unwrap();
                if unified_tuple != reps[j] {
                    let old_rep = std::mem::replace(&mut reps[j], unified_tuple);
                    let mut inc_count = 0usize;
                    for i in 0..k {
                        if old_rep[i].is_none() && reps[j][i].is_some() {
                            inc_count += 1;
                            index.on_rep_coordinate_becomes_some(j, i, reps[j][i].unwrap());
                        }
                    }
                    if inc_count > 0 {
                        let old_spec = rep_spec[j];
                        rep_spec[j] += inc_count;
                        index.on_rep_spec_increase(j, old_spec, rep_spec[j]);
                    }
                    if !work_set.contains(&j) {
                        work_queue.push_back(j);
                        work_set.insert(j);
                    }
                }
                image.insert(x, j);
            } else {
                let new_id = reps.len();
                let spec = x.iter().filter(|opt| opt.is_some()).count();
                reps.push(x.clone());
                rep_spec.push(spec);
                image.insert(x, new_id);
                work_queue.push_back(new_id);
                work_set.insert(new_id);
                index.on_new_rep(new_id, &reps[new_id], spec);
                if VERBOSE && reps.len() % 100 == 0 {
                    println!(
                        " -> Representative count reached {}. |Work|={}",
                        reps.len(),
                        work_queue.len()
                    );
                }
            }
        }
    }

    if VERBOSE {
        println!(
            "Greedy synthesis finished. Total reps: {}, total image size: {}.",
            reps.len(),
            image.len()
        );
    }
    Solution { reps, image }
}

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
        assert!(is_less_or_equal(&vec![None, None], &vec![Some(1), Some(2)]));
        assert!(is_less_or_equal(&vec![Some(1), None], &vec![Some(1), Some(2)]));
        assert!(is_less_or_equal(&vec![None, Some(2)], &vec![Some(1), Some(2)]));
        assert!(is_less_or_equal(&vec![Some(1), Some(2)], &vec![Some(1), Some(2)]));

        assert!(!is_less_or_equal(
            &vec![Some(1), Some(2)],
            &vec![Some(1), None]
        ));
        assert!(!is_less_or_equal(
            &vec![Some(1), Some(2)],
            &vec![Some(1), Some(3)]
        ));
        assert!(!is_less_or_equal(
            &vec![Some(1), Some(2)],
            &vec![None, Some(2)]
        ));
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
    fn test_simple_merge_greedy_wrapper() {
        let comp0 = vec![BTreeMap::from([(0usize, 0usize)])];
        let comp1 = vec![BTreeMap::from([(1usize, 0usize)])];
        let components = vec![comp0, comp1];
        let alphabet_size = 3;
        let start_tuple = vec![Some(0), Some(0)];

        let (states, point_map) = merge_and_build_automaton(start_tuple, &components, alphabet_size);

        assert_eq!(states.len(), 1);
        assert_eq!(states[0], vec![Some(0), Some(0)]);

        let succ0 = successor_tuple(&states[0], 0, &components);
        let succ1 = successor_tuple(&states[0], 1, &components);
        assert_eq!(*point_map.get(&succ0).unwrap(), 0);
        assert_eq!(*point_map.get(&succ1).unwrap(), 0);
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

        let sol = synthesize_greedy(&inst);
        sol.verify(&inst).expect("greedy solution must satisfy constraints");

        assert_eq!(sol.reps.len(), 2);
        assert_eq!(sol.reps[0], vec![Some(0), Some(0)]);
        assert_eq!(sol.reps[1], vec![Some(1), Some(1)]);
    }
}
