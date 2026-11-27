//! Necessary positive-read context pruning in the native weight representation.
//!
//! This kernel preserves state identities and every explicit label guard. The
//! caller must supply a separately effect-certified overapproximation of its
//! parser stacks. Shape checks here establish suffix/root dominance only; they
//! do not manufacture a grammar reachability certificate.
use super::CheckedNativeTopology;
use super::{FastBoundaryNwaState, DEFAULT_LABEL};

#[derive(Debug)]
pub struct FiniteParserReadSupport {
    states: usize,
    root: usize,
    words: usize,
    target: Vec<u32>,
    allowed: Vec<u64>,
}

impl FiniteParserReadSupport {
    pub(super) fn with_pop_classes(&self,classes:&crate::pop_classes::PopLabelClasses)->Option<Self> {
        if self.alphabet()!=classes.symbol_count() as usize {return None;}
        let alphabet=self.alphabet().checked_add(classes.len())?;
        if alphabet>50_000 || alphabet.checked_mul(self.words)?>8_000_000 {return None;}
        let mut target=self.target.clone(); let mut allowed=self.allowed.clone();
        for index in 0..classes.len() {
            let label=DEFAULT_LABEL-1-index as i32;
            let mut union=vec![0u64;self.words];
            for symbol in classes.matching_symbols(label) {
                let begin=symbol as usize*self.words;
                for (out,bits) in union.iter_mut().zip(&self.allowed[begin..begin+self.words]) {*out|=*bits;}
            }
            // Matching concrete labels may lead to several predecessor
            // residuals. Forgetting to the certified free-first-symbol root
            // is conservative, preserving every possible suffix context.
            target.push(self.root as u32); allowed.extend(union);
        }
        Some(Self {states:self.states,root:self.root,words:self.words,target,allowed})
    }

    pub(super) fn alphabet(&self) -> usize { self.target.len() }
    /// Rows are deterministic, epsilon-free adjacency residuals. Every label
    /// has one residual target independent of the source row. `live` records
    /// coaccessibility to a complete domain word. The caller retains the
    /// independent parser-effect closure certificate for these rows.
    pub fn new_checked(
        alphabet: usize, root: usize, rows: &[Vec<(u32,u32)>],
        live: &[bool], root_accepting: bool,
    ) -> Option<Self> {
        let states=rows.len();
        if states==0 || states>4096 || root>=states || alphabet==0
            || alphabet>50_000 || live.len()!=states || !root_accepting || !live[root] {
            return None;
        }
        let words=states.div_ceil(64);
        let mut target=vec![u32::MAX;alphabet];
        let mut allowed=vec![0u64;alphabet.checked_mul(words)?];
        for (source,row) in rows.iter().enumerate() {
            let mut previous=None;
            for &(label,next) in row {
                if label as usize>=alphabet || next as usize>=states
                    || previous.is_some_and(|p|p>=label) { return None; }
                previous=Some(label);
                let prior=&mut target[label as usize];
                if *prior!=u32::MAX && *prior!=next { return None; }
                *prior=next;
                if live[next as usize] {
                    allowed[label as usize*words+source/64]|=1u64<<(source%64);
                }
            }
        }
        for label in 0..alphabet {
            if target[label]!=u32::MAX && live[target[label] as usize]
                && !has(&allowed[label*words..(label+1)*words],root) { return None; }
        }
        Some(Self{states,root,words,target,allowed})
    }
}

#[derive(Debug,Default)]
pub(super) struct ReadSupportProfile {
    states:usize,
    domain_states:usize,
    live_states:usize,
    live_edges_before:usize,
    live_edges_after:usize,
    blocked_labels:usize,
    forgotten_defaults:usize,
    propagation_words:usize,
}

fn has(bits:&[u64],id:usize)->bool {bits[id/64]&(1u64<<(id%64))!=0}
fn insert(bits:&mut[u64],id:usize,root:usize){
    if has(bits,root){return;}
    if id==root{bits.fill(0);}
    bits[id/64]|=1u64<<(id%64);
}

/// Sparse storage for the same transfer. An offset exists iff a successful
/// incoming transfer supplied a nonempty context. Original graph IDs remain
/// unchanged; only the storage of context bitsets is allocated on demand.
pub(super) struct ReadContextTransfer<'a> {
    domain: &'a FiniteParserReadSupport,
    offsets: Vec<u32>,
    contexts: Vec<u64>,
    source: Vec<u64>,
    source_nonempty: bool,
    word_limit: usize,
    dense: bool,
    propagation_words: usize,
}

impl<'a> ReadContextTransfer<'a> {
    pub fn new(domain: &'a FiniteParserReadSupport, nodes: usize, starts: &[u32]) -> Option<Self> {
        Self::with_word_limit(domain, nodes, starts, 8_000_000)
    }

    fn with_word_limit(domain: &'a FiniteParserReadSupport, nodes: usize, starts: &[u32], limit: usize) -> Option<Self> {
        if nodes == 0 || nodes > 200_000 || starts.iter().any(|&q| q as usize >= nodes) { return None; }
        let mut result = Self { domain, offsets: vec![u32::MAX; nodes], contexts: Vec::new(),
            source: vec![0; domain.words], source_nonempty: false, word_limit: limit.min(8_000_000), dense: false, propagation_words: 0 };
        for &q in starts {
            let begin = result.destination(q as usize)?;
            insert(&mut result.contexts[begin..begin + domain.words], domain.root, domain.root);
        }
        Some(result)
    }

    /// Dense preallocation for callers that already materialize every row.
    /// The same offset sentinel records nonempty support in either layout.
    fn new_dense(domain: &'a FiniteParserReadSupport, nodes: usize, starts: &[u32]) -> Option<Self> {
        if nodes == 0 || nodes > 200_000 || nodes.checked_mul(domain.words)? > 8_000_000 {
            return None;
        }
        let mut result = Self {
            domain,
            offsets: vec![u32::MAX; nodes],
            contexts: vec![0; nodes * domain.words],
            source: vec![0; domain.words],
            source_nonempty: false,
            word_limit: 8_000_000,
            dense: true,
            propagation_words: 0,
        };
        for &state in starts {
            let begin = result.destination(state as usize)?;
            insert(&mut result.contexts[begin..begin + domain.words], domain.root, domain.root);
        }
        Some(result)
    }

    fn destination(&mut self, target: usize) -> Option<usize> {
        let offset = *self.offsets.get(target)?;
        if offset != u32::MAX { return Some(offset as usize); }
        if self.dense {
            let begin = target * self.domain.words;
            self.offsets[target] = begin as u32;
            return Some(begin);
        }
        let begin = self.contexts.len();
        let end = begin.checked_add(self.domain.words)?;
        if end > self.word_limit { return None; }
        self.contexts.resize(end, 0);
        self.offsets[target] = begin as u32;
        Some(begin)
    }

    pub fn allocated_nodes(&self) -> usize { self.contexts.len() / self.domain.words }

    pub fn state_nonempty(&self, q: usize) -> Option<bool> {
        Some(*self.offsets.get(q)? != u32::MAX)
    }

    pub fn load_source(&mut self, q: usize) -> Option<bool> {
        let offset = *self.offsets.get(q)?;
        self.source_nonempty = offset != u32::MAX;
        if self.source_nonempty {
            let begin = offset as usize;
            self.source.copy_from_slice(&self.contexts[begin..begin + self.domain.words]);
            debug_assert!(self.source.iter().any(|&bits| bits != 0));
        }
        Some(self.source_nonempty)
    }

    pub fn read_allowed(&self, label: i32) -> Option<bool> {
        if label != DEFAULT_LABEL && (label < 0 || label as usize >= self.domain.target.len()) { return None; }
        if !self.source_nonempty { return Some(false); }
        if label == DEFAULT_LABEL { return Some(true); }
        let begin = label as usize * self.domain.words;
        Some(self.source.iter().zip(&self.domain.allowed[begin..begin + self.domain.words])
            .any(|(a, b)| a & b != 0))
    }

    pub fn transfer_epsilon(&mut self, target: usize) -> Option<()> {
        if target >= self.offsets.len() { return None; }
        if !self.source_nonempty { return Some(()); }
        let root = self.domain.root;
        let source_root = has(&self.source, root);
        let begin = self.destination(target)?;
        let dest = &mut self.contexts[begin..begin + self.domain.words];
        if source_root { insert(dest, root, root); }
        else if !has(dest, root) {
            for (a, b) in dest.iter_mut().zip(&self.source) { *a |= *b; }
            self.propagation_words += self.domain.words;
        }
        Some(())
    }

    /// Called only after read_allowed(label) was true for the loaded source.
    pub fn transfer_read(&mut self, label: i32, target: usize) -> Option<()> {
        if target >= self.offsets.len() { return None; }
        if !self.source_nonempty { return Some(()); }
        let context = if label == DEFAULT_LABEL { self.domain.root }
            else { *self.domain.target.get(label as usize)? as usize };
        if context >= self.domain.states { return None; }
        let begin = self.destination(target)?;
        insert(&mut self.contexts[begin..begin + self.domain.words], context, self.domain.root);
        Some(())
    }
}


pub(super) fn restrict(
    nwa:&mut [FastBoundaryNwaState], starts:&[u32], domain:&FiniteParserReadSupport,
) -> Option<ReadSupportProfile> {
    restrict_with_topology(nwa, starts, domain, None)
}

pub(super) fn restrict_with_topology(
    nwa:&mut [FastBoundaryNwaState], starts:&[u32], domain:&FiniteParserReadSupport,
    topology:Option<&CheckedNativeTopology>,
) -> Option<ReadSupportProfile> {
    let n = nwa.len();
    let mut transfer = ReadContextTransfer::new_dense(domain, n, starts)?;
    if topology.is_some_and(|order| order.order.len() != n) {
        return None;
    }
    let mut edges = 0usize;
    // Validate the complete graph, including labels/targets of dead guards,
    // before changing any coefficients.
    for (state_id, state) in nwa.iter().enumerate() {
        for (label, branches) in &state.transitions {
            if *label != DEFAULT_LABEL && (*label < 0 || *label as usize >= domain.target.len()) {
                return None;
            }
            for &(target, weight) in branches {
                if target as usize >= n {
                    return None;
                }
                if weight != 0 {
                    if topology.is_some_and(|order| !order.permits(state_id, target)) {
                        return None;
                    }
                    edges += 1;
                }
            }
        }
        for &(target, weight) in &state.epsilons {
            if target as usize >= n {
                return None;
            }
            if weight != 0 {
                if topology.is_some_and(|order| !order.permits(state_id, target)) {
                    return None;
                }
                edges += 1;
            }
        }
    }
    if edges > 4_000_000 {
        return None;
    }
    let owned_order;
    let order = if let Some(topology) = topology {
        topology.order.as_slice()
    } else {
        owned_order = super::finite_signed_graph::topological_order(nwa as &[FastBoundaryNwaState])?;
        owned_order.as_slice()
    };
    let mut profile = ReadSupportProfile {
        states: n, domain_states: domain.states, live_edges_before: edges,
        ..Default::default()
    };
    for &state_id in order {
        let state = &mut nwa[state_id as usize];
        if !transfer.load_source(state_id as usize)? {
            state.final_weight = 0;
            for (_, weight) in state.epsilons.iter_mut()
                .chain(state.transitions.iter_mut().flat_map(|(_, branches)| branches.iter_mut()))
            {
                *weight = 0;
            }
            continue;
        }
        profile.live_states += 1;
        for &(target, weight) in &state.epsilons {
            if weight != 0 {
                transfer.transfer_epsilon(target as usize)?;
                profile.live_edges_after += 1;
            }
        }
        for (label, branches) in &mut state.transitions {
            if !transfer.read_allowed(*label)? {
                profile.blocked_labels += 1;
                for (_, weight) in branches {
                    *weight = 0;
                }
                continue;
            }
            for &(target, weight) in branches.iter() {
                if weight != 0 {
                    transfer.transfer_read(*label, target as usize)?;
                    profile.live_edges_after += 1;
                    profile.forgotten_defaults += usize::from(*label == DEFAULT_LABEL);
                }
            }
        }
    }
    profile.propagation_words = transfer.propagation_words;
    Some(profile)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn state(final_weight:u32, edges:Vec<(i32,u32,u32)>)->FastBoundaryNwaState {
        FastBoundaryNwaState{final_weight,epsilons:vec![],transitions:edges.into_iter()
            .map(|(label,target,weight)|(label,smallvec::smallvec![(target,weight)])).collect()}
    }
    #[test]
    fn native_context_preserves_empty_guards_and_default_forgetting(){
        let domain=FiniteParserReadSupport::new_checked(3,0,
            &[vec![(0,1),(1,1),(2,2)],vec![(0,1)],vec![(2,2)]],&[true,true,true],true).unwrap();
        let mut nwa=vec![state(0,vec![(1,1,1)]),state(0,vec![(0,3,1),(2,2,1),(DEFAULT_LABEL,4,1)]),
            state(1,vec![]),state(1,vec![]),state(0,vec![(2,5,1)]),state(1,vec![])];
        let stats=restrict(&mut nwa,&[0],&domain).unwrap();
        assert_eq!(stats.blocked_labels,1);
        assert_eq!(nwa[1].transitions[1].0,2);
        assert_eq!(nwa[1].transitions[1].1[0].1,0);
        assert_eq!(nwa[2].final_weight,0);
        assert_eq!(nwa[3].final_weight,1);
        assert_eq!(nwa[4].transitions[0].1[0].1,1);
        assert_eq!(nwa[5].final_weight,1);
    }
    #[test]
    fn malformed_context_and_cyclic_input_decline(){
        assert!(FiniteParserReadSupport::new_checked(2,0,&[vec![(0,0)],vec![(1,1)]],&[true,true],true).is_none());
        assert!(FiniteParserReadSupport::new_checked(2,0,&[vec![(0,0),(0,1)],vec![]],&[true,true],true).is_none());
        let domain=FiniteParserReadSupport::new_checked(1,0,&[vec![(0,0)]],&[true],true).unwrap();
        let mut nwa=vec![state(1,vec![(0,0,1)])];
        assert!(restrict(&mut nwa,&[0],&domain).is_none());
        assert_eq!(nwa[0].final_weight,1);
    }
    #[test]
    fn sparse_context_allocates_only_nonempty_blocks_and_declines_before_publication(){
        let domain=FiniteParserReadSupport::new_checked(3,0,
            &[vec![(0,1),(1,1),(2,2)],vec![(0,1)],vec![(2,2)]],&[true,true,true],true).unwrap();
        let mut c=ReadContextTransfer::with_word_limit(&domain,100,&[0],3).unwrap();
        assert_eq!(c.allocated_nodes(),1);
        assert!(!c.load_source(90).unwrap());
        assert!(!c.read_allowed(DEFAULT_LABEL).unwrap());
        c.transfer_epsilon(91).unwrap();
        assert!(!c.state_nonempty(91).unwrap());
        assert!(c.load_source(0).unwrap());
        c.transfer_read(1,50).unwrap();
        assert!(c.load_source(50).unwrap());
        assert!(!c.read_allowed(2).unwrap());
        c.transfer_read(DEFAULT_LABEL,70).unwrap();
        assert_eq!(c.allocated_nodes(),3);
        assert!(c.load_source(70).unwrap());
        assert!(c.read_allowed(2).unwrap());
        assert!(c.transfer_epsilon(99).is_none());
        assert!(!c.state_nonempty(99).unwrap(),"failed storage growth cannot publish a node");
        c.transfer_epsilon(50).unwrap();
        assert!(c.load_source(50).unwrap());
        assert!(c.read_allowed(2).unwrap(),"root dominates an existing narrower row");
        assert!(c.load_source(100).is_none());
        assert!(c.read_allowed(-1).is_none());
        assert_eq!(c.allocated_nodes(),3);
    }

}
