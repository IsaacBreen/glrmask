#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]


use std::collections::{BTreeMap, HashSet, VecDeque};

use crate::ds::weight::Weight;


pub type Label = i32;


#[derive(Debug, Clone, Default)]
pub struct NWAState {
    
    
    pub final_weight: Option<Weight>,
    
    pub transitions: BTreeMap<Label, Vec<(u32, Weight)>>,
    
    pub epsilons: Vec<(u32, Weight)>,
}


#[derive(Debug, Clone)]
pub struct NWA {
    
    pub states: Vec<NWAState>,
    
    pub start_states: Vec<u32>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NwaBody {
    pub start_states: Vec<u32>,
}

impl NwaBody {
    pub fn union(left: &Self, right: &Self) -> Self {
        let mut start_states = left.start_states.clone();
        start_states.extend(right.start_states.iter().copied());
        Self { start_states }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NwaTraversalData {
    pub comp_id: Vec<usize>,
    pub sccs: Vec<Vec<usize>>,
    pub topo: Vec<usize>,
}

impl NWA {
    
    pub fn new(num_tsids: u32, max_token: u32) -> Self {
        let _ = (num_tsids, max_token);
        Self {
            states: Vec::new(),
            start_states: Vec::new(),
        }
    }

    
    pub fn add_state(&mut self) -> u32 {
        let id = self.states.len() as u32;
        self.states.push(NWAState::default());
        id
    }

    
    pub fn num_states(&self) -> u32 {
        self.states.len() as u32
    }

    
    pub fn set_final_weight(&mut self, state: u32, weight: Weight) {
        if let Some(entry) = self.states.get_mut(state as usize) {
            entry.final_weight = Some(weight);
        }
    }

    
    pub fn add_transition(&mut self, from: u32, label: Label, to: u32, weight: Weight) {
        if let Some(entry) = self.states.get_mut(from as usize) {
            entry.transitions.entry(label).or_default().push((to, weight));
        }
    }

    
    pub fn add_epsilon(&mut self, from: u32, to: u32, weight: Weight) {
        if let Some(entry) = self.states.get_mut(from as usize) {
            entry.epsilons.push((to, weight));
        }
    }

    
    pub fn num_transitions(&self) -> usize {
        self.states
            .iter()
            .map(|state| state.epsilons.len() + state.transitions.values().map(Vec::len).sum::<usize>())
            .sum()
    }

    pub fn body(&self) -> NwaBody {
        NwaBody {
            start_states: self.start_states.clone(),
        }
    }

    pub fn append_with_body(&mut self, other: &NWA) -> NwaBody {
        let offset = self.states.len() as u32;
        for _ in &other.states {
            self.add_state();
        }

        for (state_id, state) in other.states.iter().enumerate() {
            let dst_state = offset + state_id as u32;
            if let Some(final_weight) = state.final_weight.clone() {
                self.set_final_weight(dst_state, final_weight);
            }
            for (&label, targets) in &state.transitions {
                for (target, weight) in targets {
                    self.add_transition(dst_state, label, offset + *target, weight.clone());
                }
            }
            for (target, weight) in &state.epsilons {
                self.add_epsilon(dst_state, offset + *target, weight.clone());
            }
        }

        NwaBody {
            start_states: other.start_states.iter().map(|state| offset + *state).collect(),
        }
    }

    pub fn concatenate_in_place(&mut self, left: &NWA, right_body: &NwaBody) -> NwaBody {
        let offset = self.states.len() as u32;
        let left_body = self.append_with_body(left);

        for state_id in offset as usize..(offset as usize + left.states.len()) {
            if let Some(final_weight) = self.states[state_id].final_weight.take() {
                if !final_weight.is_empty() {
                    for &right_start in &right_body.start_states {
                        self.add_epsilon(state_id as u32, right_start, final_weight.clone());
                    }
                }
            }
        }

        left_body
    }

    pub fn union_in_place(&mut self, other: &NWA, existing_body: &NwaBody) -> NwaBody {
        let other_body = self.append_with_body(other);
        NwaBody::union(existing_body, &other_body)
    }

    pub fn reverse(&self) -> Self {
        let mut rev = Self {
            states: vec![NWAState::default(); self.states.len()],
            start_states: Vec::new(),
        };

        let super_start = rev.add_state();
        rev.start_states.push(super_start);

        for (src, state) in self.states.iter().enumerate() {
            for (&label, targets) in &state.transitions {
                for (dst, weight) in targets {
                    rev.add_transition(*dst, label, src as u32, weight.clone());
                }
            }
            for (dst, weight) in &state.epsilons {
                rev.add_epsilon(*dst, src as u32, weight.clone());
            }
            if let Some(final_weight) = &state.final_weight {
                if !final_weight.is_empty() {
                    rev.add_epsilon(super_start, src as u32, final_weight.clone());
                }
            }
        }

        for &start in &self.start_states {
            if let Some(state) = rev.states.get_mut(start as usize) {
                let updated = match &state.final_weight {
                    Some(existing) => existing.union(&Weight::all()),
                    None => Weight::all(),
                };
                state.final_weight = Some(updated);
            }
        }

        rev
    }

    pub fn compute_traversal_data(&self) -> NwaTraversalData {
        let (sccs, comp_id) = compute_sccs(self);
        let mut scc_adj = vec![HashSet::new(); sccs.len()];
        let mut indeg = vec![0usize; sccs.len()];

        for (src, state) in self.states.iter().enumerate() {
            let src_comp = comp_id[src];
            let mut neighbors = Vec::new();
            for targets in state.transitions.values() {
                for (dst, _) in targets {
                    neighbors.push(*dst as usize);
                }
            }
            for (dst, _) in &state.epsilons {
                neighbors.push(*dst as usize);
            }

            for dst in neighbors {
                let dst_comp = comp_id[dst];
                if src_comp != dst_comp && scc_adj[src_comp].insert(dst_comp) {
                    indeg[dst_comp] += 1;
                }
            }
        }

        let mut topo = Vec::with_capacity(sccs.len());
        let mut queue = VecDeque::new();
        for (comp, degree) in indeg.iter().enumerate() {
            if *degree == 0 {
                queue.push_back(comp);
            }
        }

        while let Some(comp) = queue.pop_front() {
            topo.push(comp);
            for &next in &scc_adj[comp] {
                indeg[next] -= 1;
                if indeg[next] == 0 {
                    queue.push_back(next);
                }
            }
        }

        NwaTraversalData { comp_id, sccs, topo }
    }

    
    pub fn is_acyclic(&self) -> bool {
        let num_states = self.states.len();

        for (state_id, state) in self.states.iter().enumerate() {
            if state
                .transitions
                .values()
                .flatten()
                .any(|(target, _)| *target as usize == state_id)
            {
                return false;
            }
            if state
                .epsilons
                .iter()
                .any(|(target, _)| *target as usize == state_id)
            {
                return false;
            }
        }

        fn visit(state_id: usize, states: &[NWAState], colors: &mut [u8]) -> bool {
            colors[state_id] = 1;

            for (target, _) in states[state_id].transitions.values().flatten() {
                let target = *target as usize;
                if target >= colors.len() {
                    continue;
                }
                match colors[target] {
                    1 => return false,
                    0 => {
                        if !visit(target, states, colors) {
                            return false;
                        }
                    }
                    _ => {}
                }
            }

            for (target, _) in &states[state_id].epsilons {
                let target = *target as usize;
                if target >= colors.len() {
                    continue;
                }
                match colors[target] {
                    1 => return false,
                    0 => {
                        if !visit(target, states, colors) {
                            return false;
                        }
                    }
                    _ => {}
                }
            }

            colors[state_id] = 2;
            true
        }

        let mut colors = vec![0u8; num_states];
        for state_id in 0..num_states {
            if colors[state_id] == 0 && !visit(state_id, &self.states, &mut colors) {
                return false;
            }
        }
        true
    }

    
    pub fn max_position(&self) -> u32 {
        self.states.len().saturating_sub(1) as u32
    }

    
    
    
    pub fn display_with_symbols<'a>(
        &'a self,
        symbols: &'a std::collections::BTreeMap<Label, String>,
    ) -> NWADisplayWithSymbols<'a> {
        NWADisplayWithSymbols { nwa: self, symbols }
    }

    
    
    pub fn display_with_all_maps<'a>(
        &'a self,
        symbols: &'a std::collections::BTreeMap<Label, String>,
        tsid_names: &'a std::collections::BTreeMap<u32, String>,
        token_names: &'a std::collections::BTreeMap<u32, String>,
    ) -> NWADisplayWithAllMaps<'a> {
        NWADisplayWithAllMaps {
            nwa: self,
            symbols,
            tsid_names,
            token_names,
        }
    }
}

fn compute_sccs(nwa: &NWA) -> (Vec<Vec<usize>>, Vec<usize>) {
    let num_states = nwa.states.len();
    let mut adj = vec![Vec::new(); num_states];
    let mut rev_adj = vec![Vec::new(); num_states];

    for (src, state) in nwa.states.iter().enumerate() {
        let mut neighbors = Vec::new();
        for targets in state.transitions.values() {
            for (dst, _) in targets {
                neighbors.push(*dst as usize);
            }
        }
        for (dst, _) in &state.epsilons {
            neighbors.push(*dst as usize);
        }

        for dst in neighbors {
            adj[src].push(dst);
            rev_adj[dst].push(src);
        }
    }

    let mut order = Vec::new();
    let mut visited = vec![false; num_states];
    for start in 0..num_states {
        if visited[start] {
            continue;
        }
        let mut stack = vec![(start, false)];
        while let Some((state, processed)) = stack.pop() {
            if processed {
                order.push(state);
                continue;
            }
            if visited[state] {
                continue;
            }
            visited[state] = true;
            stack.push((state, true));
            for &next in &adj[state] {
                if !visited[next] {
                    stack.push((next, false));
                }
            }
        }
    }

    let mut comp_id = vec![usize::MAX; num_states];
    let mut sccs = Vec::new();
    for &start in order.iter().rev() {
        if comp_id[start] != usize::MAX {
            continue;
        }
        let comp = sccs.len();
        let mut stack = vec![start];
        comp_id[start] = comp;
        let mut component = Vec::new();
        while let Some(state) = stack.pop() {
            component.push(state);
            for &prev in &rev_adj[state] {
                if comp_id[prev] == usize::MAX {
                    comp_id[prev] = comp;
                    stack.push(prev);
                }
            }
        }
        sccs.push(component);
    }

    (sccs, comp_id)
}


fn fmt_nwa_states(
    nwa: &NWA,
    f: &mut std::fmt::Formatter<'_>,
    label_fn: &dyn Fn(Label) -> String,
    weight_fn: &dyn Fn(&Weight) -> String,
) -> std::fmt::Result {
    let start_set: std::collections::BTreeSet<u32> = nwa.start_states.iter().copied().collect();

    for (i, st) in nwa.states.iter().enumerate() {
        if st.transitions.is_empty() && st.epsilons.is_empty() && st.final_weight.is_none() {
            continue;
        }

        
        let start_mark = if start_set.contains(&(i as u32)) { " [START]" } else { "" };
        writeln!(f, "  State {i}{start_mark}")?;

        
        if let Some(w) = &st.final_weight {
            writeln!(f, "    final: {}", weight_fn(w))?;
        }

        
        for (label, targets) in &st.transitions {
            let lbl = label_fn(*label);
            for (tgt, w) in targets {
                writeln!(f, "    {lbl} → State {tgt}")?;
                writeln!(f, "      weight: {}", weight_fn(w))?;
            }
        }

        
        for (tgt, w) in &st.epsilons {
            writeln!(f, "    ε → State {tgt}")?;
            writeln!(f, "      weight: {}", weight_fn(w))?;
        }
    }
    Ok(())
}


impl std::fmt::Display for NWA {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "NWA: {} states, start={:?}", self.states.len(), self.start_states)?;
        fmt_nwa_states(self, f, &|l| l.to_string(), &|w| format!("{w}"))
    }
}


pub struct NWADisplayWithSymbols<'a> {
    nwa: &'a NWA,
    symbols: &'a std::collections::BTreeMap<Label, String>,
}

impl std::fmt::Display for NWADisplayWithSymbols<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let nwa = self.nwa;
        writeln!(f, "NWA: {} states, start={:?}", nwa.states.len(), nwa.start_states)?;
        let syms = self.symbols;
        fmt_nwa_states(nwa, f,
            &|label| match syms.get(&label) {
                Some(name) => name.clone(),
                None => format!("{label}"),
            },
            &|w| format!("{w}"),
        )
    }
}


pub struct NWADisplayWithAllMaps<'a> {
    nwa: &'a NWA,
    symbols: &'a std::collections::BTreeMap<Label, String>,
    tsid_names: &'a std::collections::BTreeMap<u32, String>,
    token_names: &'a std::collections::BTreeMap<u32, String>,
}

impl std::fmt::Display for NWADisplayWithAllMaps<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let nwa = self.nwa;
        let syms = self.symbols;
        writeln!(f, "NWA: {} states, start={:?}", nwa.states.len(), nwa.start_states)?;
        fmt_nwa_states(
            nwa,
            f,
            &|label| syms.get(&label).cloned().unwrap_or_else(|| label.to_string()),
            &|weight| format!("{}", weight.display_with_names(self.tsid_names, self.token_names)),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use range_set_blaze::RangeSetBlaze;

    #[test]
    fn test_nwa_basic() {
        let mut nwa = NWA::new(2, 10);
        let s0 = nwa.add_state();
        let s1 = nwa.add_state();
        let s2 = nwa.add_state();

        let w = Weight::all();
        nwa.add_transition(s0, 0, s1, w.clone());
        nwa.add_epsilon(s1, s2, w.clone());
        nwa.set_final_weight(s2, w);

        assert_eq!(nwa.num_states(), 3);
        assert_eq!(nwa.num_transitions(), 2);
        assert!(nwa.states[s2 as usize].final_weight.is_some());
    }
}
