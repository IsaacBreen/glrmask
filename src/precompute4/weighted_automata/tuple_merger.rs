#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};

pub type ProductTuple = Vec<Option<usize>>;

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

pub fn successor_tuple(
    tuple: &ProductTuple,
    symbol: usize,
    components: &[Vec<BTreeMap<usize, usize>>],
) -> ProductTuple {
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
    alphabet_size: usize,
) -> (Vec<ProductTuple>, HashMap<ProductTuple, usize>) {
    let mut states: Vec<ProductTuple> = Vec::new();
    let mut point_map: HashMap<ProductTuple, usize> = HashMap::new();
    let mut worklist: VecDeque<usize> = VecDeque::new();

    // The "other" symbol, representing default transitions, is conventionally the largest symbol index.
    let other_index = if alphabet_size > 0 { alphabet_size - 1 } else { 0 };

    let start_id = 0;
    states.push(start_tuple.clone());
    point_map.insert(start_tuple, start_id);
    worklist.push_back(start_id);

    while let Some(state_id) = worklist.pop_front() {
        let representative = states[state_id].clone();

        let mut alphabet = BTreeSet::new();
        for (i, comp_state_opt) in representative.iter().enumerate() {
            if let Some(comp_state) = comp_state_opt {
                for &symbol in components[i][*comp_state].keys() {
                    alphabet.insert(symbol);
                }
            }
        }
        // The default transition must always be explored to ensure the product automaton is complete.
        alphabet.insert(other_index);

        for symbol in alphabet {

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

    (states, point_map)
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

        let (states, point_map) = merge_and_build_automaton(start_tuple, &components, 3);

        assert_eq!(states.len(), 1);
        assert_eq!(states[0], vec![Some(0), Some(0)]);

        let succ0 = successor_tuple(&states[0], 0, &components);
        let succ1 = successor_tuple(&states[0], 1, &components);
        assert_eq!(*point_map.get(&succ0).unwrap(), 0);
        assert_eq!(*point_map.get(&succ1).unwrap(), 0);
    }
}
