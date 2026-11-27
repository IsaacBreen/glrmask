//! Sink-first component condensation and monotone reachability.
use crate::ds::bitset::BitSet;

pub(super) struct Components {
    pub(super) of: Vec<usize>,
    pub(super) count: usize,
}

/// Iterative Tarjan: each cross-component edge points to a lower ID.
/// Edges are read directly from the caller's immutable graph.
pub(super) fn components<I: Iterator<Item = usize>>(
    n: usize, edges: impl Fn(usize) -> I,
) -> Components {
    let mut of = vec![usize::MAX; n];
    let mut discovery = vec![usize::MAX; n];
    let mut low = vec![0; n];
    let mut pending = Vec::new();
    let mut dfs = Vec::new();
    let mut index = 0;
    let mut count = 0;
    for root in 0..n {
        if discovery[root] != usize::MAX { continue; }
        discovery[root] = index;
        low[root] = index;
        index += 1;
        pending.push(root);
        dfs.push((root, edges(root)));
        while let Some((state, children)) = dfs.last_mut() {
            let state = *state;
            if let Some(target) = children.next() {
                if discovery[target] == usize::MAX {
                    discovery[target] = index;
                    low[target] = index;
                    index += 1;
                    pending.push(target);
                    dfs.push((target, edges(target)));
                } else if of[target] == usize::MAX {
                    low[state] = low[state].min(discovery[target]);
                }
            } else {
                if low[state] == discovery[state] {
                    loop {
                        let member = pending.pop().expect("active SCC member");
                        of[member] = count;
                        if member == state { break; }
                    }
                    count += 1;
                }
                dfs.pop();
                if let Some((parent, _)) = dfs.last() {
                    low[*parent] = low[*parent].min(low[state]);
                }
            }
        }
    }
    Components { of, count }
}

impl Components {
    pub(super) fn successors<I: Iterator<Item = usize>>(
        &self, edges: impl Fn(usize) -> I,
    ) -> Vec<Vec<usize>> {
        let mut successors = vec![Vec::new(); self.count];
        for (source, &class) in self.of.iter().enumerate() {
            for target in edges(source) {
                let child = self.of[target];
                if child != class {
                    debug_assert!(child < class);
                    successors[class].push(child);
                }
            }
        }
        for row in &mut successors {
            row.sort_unstable();
            row.dedup();
        }
        successors
    }
}

/// Inclusive reachability is the union of local seeds and all successor
/// components. No repeated fixed-point scans or per-state epsilon closures.
pub(super) fn reachable_groups<I: Iterator<Item = usize>>(
    n: usize, groups: usize, edges: impl Fn(usize) -> I,
    mut seed: impl FnMut(usize, &mut BitSet),
) -> Vec<BitSet> {
    if groups == 0 { return vec![BitSet::new(0); n]; }
    let components = components(n, &edges);
    let successors = components.successors(edges);
    let mut rows = vec![BitSet::new(groups); components.count];
    for (state, &class) in components.of.iter().enumerate() {
        seed(state, &mut rows[class]);
    }
    for (class, children) in successors.into_iter().enumerate() {
        let (previous, current) = rows.split_at_mut(class);
        for child in children {
            current[0].union_with(&previous[child]);
        }
    }
    components.of.into_iter().map(|class| rows[class].clone()).collect()
}
