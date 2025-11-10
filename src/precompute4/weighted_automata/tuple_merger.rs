#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};

pub type ProductTuple = Vec<Option<usize>>;
pub type MergedState = (ProductTuple, BTreeMap<usize, usize>);

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

pub fn successor_tuple(tuple: &ProductTuple, symbol: usize, components: &[Vec<BTreeMap<usize, usize>>]) -> ProductTuple {
    let k = components.len();
    let mut out = Vec::with_capacity(k);
    for i in 0..k {
        match tuple[i] {
            Some(s) => {
                if let Some(&v) = components[i][s].get(&symbol) {
                    out.push(Some(v));
                } else {
                    out.push(None);
                }
            }
            None => {
                out.push(None);
            }
        }
    }
    out
}

pub fn merge_and_build_automaton(
    start_tuple: ProductTuple,
    components: &[Vec<BTreeMap<usize, usize>>],
) -> Vec<MergedState> {
    let mut states: Vec<ProductTuple> = Vec::new();
    let mut point_map: HashMap<ProductTuple, usize> = HashMap::new();
    let mut worklist: VecDeque<usize> = VecDeque::new();

    let start_id = 0;
    states.push(start_tuple.clone());
    point_map.insert(start_tuple, start_id);
    worklist.push_back(start_id);

    let mut alphabet = BTreeSet::new();
    for comp in components {
        for state_trans in comp {
            for &symbol in state_trans.keys() {
                alphabet.insert(symbol);
            }
        }
    }

    while let Some(state_id) = worklist.pop_front() {
        let representative = states[state_id].clone();

        for &symbol in &alphabet {
            let successor = successor_tuple(&representative, symbol, components);

            if point_map.contains_key(&successor) {
                continue;
            }

            let mut assigned_id = None;
            for id in 0..states.len() {
                if let Some(new_rep) = unify_tuples(&states[id], &successor) {
                    if new_rep != states[id] {
                        states[id] = new_rep;
                        if !worklist.contains(&id) {
                            worklist.push_back(id);
                        }
                    }
                    assigned_id = Some(id);
                    break;
                }
            }

            let home_id = assigned_id.unwrap_or_else(|| {
                let new_id = states.len();
                states.push(successor.clone());
                worklist.push_back(new_id);
                new_id
            });

            point_map.insert(successor, home_id);
        }
    }

    let mut final_states = Vec::with_capacity(states.len());
    for rep in states.iter() {
        let mut transitions = BTreeMap::new();
        for &symbol in &alphabet {
            let succ = successor_tuple(rep, symbol, components);
            let target_id = *point_map.get(&succ).expect("Successor point must have an assigned state");
            transitions.insert(symbol, target_id);
        }
        final_states.push((rep.clone(), transitions));
    }
    final_states
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unify_tuples() {
        assert_eq!(unify_tuples(&vec![Some(1), None], &vec![None, Some(2)]), Some(vec![Some(1), Some(2)]));
        assert_eq!(unify_tuples(&vec![Some(1), Some(2)], &vec![Some(1), Some(2)]), Some(vec![Some(1), Some(2)]));
        assert_eq!(unify_tuples(&vec![Some(1), None], &vec![Some(1), Some(3)]), Some(vec![Some(1), Some(3)]));
        assert_eq!(unify_tuples(&vec![Some(1), Some(2)], &vec![Some(1), Some(3)]), None);
        assert_eq!(unify_tuples(&vec![None, None], &vec![Some(1), Some(2)]), Some(vec![Some(1), Some(2)]));
    }

    #[test]
    fn test_simple_merge() {
        let comp0 = vec![BTreeMap::from([(0, 0)])];
        let comp1 = vec![BTreeMap::from([(1, 0)])];
        let components = vec![comp0, comp1];

        let start_tuple = vec![Some(0), Some(0)];

        let automaton_states = merge_and_build_automaton(start_tuple, &components);

        assert_eq!(automaton_states.len(), 1);

        let s0_id = 0;
        assert_eq!(automaton_states[s0_id].0, vec![Some(0), Some(0)]);

        assert_eq!(automaton_states[s0_id].1[&0], s0_id);
        assert_eq!(automaton_states[s0_id].1[&1], s0_id);
    }
}
