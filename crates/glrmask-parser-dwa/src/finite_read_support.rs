//! Necessary positive-read context pruning in the native weight representation.
//!
//! This kernel preserves state identities and every explicit label guard. The
//! caller must supply a separately effect-certified overapproximation of its
//! parser stacks. Shape checks here establish suffix/root dominance only; they
//! do not manufacture a grammar reachability certificate.
use super::{FastBoundaryNwaState, DEFAULT_LABEL};
use std::collections::VecDeque;

#[derive(Debug)]
pub struct FiniteParserReadSupport {
    states: usize,
    root: usize,
    words: usize,
    target: Vec<u32>,
    allowed: Vec<u64>,
}

impl FiniteParserReadSupport {
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

pub(super) fn restrict(
    nwa:&mut [FastBoundaryNwaState], starts:&[u32], domain:&FiniteParserReadSupport,
) -> Option<ReadSupportProfile> {
    let (n,words,root)=(nwa.len(),domain.words,domain.root);
    if n==0 || n>200_000 || n.checked_mul(words)?>8_000_000
        || starts.iter().any(|&q|q as usize>=n) {return None;}
    let mut indegree=vec![0usize;n];
    let mut edges=0usize;
    for state in nwa.iter() {
        for (label,branches) in &state.transitions {
            if *label!=DEFAULT_LABEL && (*label<0 || *label as usize>=domain.target.len()) {return None;}
            for &(next,weight) in branches {
                if next as usize>=n {return None;}
                if weight!=0 {indegree[next as usize]+=1;edges+=1;}
            }
        }
        for &(next,weight) in &state.epsilons {
            if next as usize>=n{return None;}
            if weight!=0 {indegree[next as usize]+=1;edges+=1;}
        }
    }
    if edges>4_000_000{return None;}
    let mut queue=(0..n).filter(|&q|indegree[q]==0).collect::<VecDeque<_>>();
    let mut topo=Vec::with_capacity(n);
    while let Some(q)=queue.pop_front(){
        topo.push(q);
        for &(next,weight) in nwa[q].epsilons.iter()
            .chain(nwa[q].transitions.iter().flat_map(|(_,branches)|branches.iter())) {
            if weight==0{continue;}
            indegree[next as usize]-=1;
            if indegree[next as usize]==0{queue.push_back(next as usize);}
        }
    }
    if topo.len()!=n{return None;}
    let mut contexts=vec![0u64;n*words];
    for &q in starts {insert(&mut contexts[q as usize*words..(q as usize+1)*words],root,root);}
    let mut source=vec![0;words];
    let mut profile=ReadSupportProfile{states:n,domain_states:domain.states,live_edges_before:edges,..Default::default()};
    for q in topo {
        source.copy_from_slice(&contexts[q*words..(q+1)*words]);
        let state=&mut nwa[q];
        if source.iter().all(|&word|word==0){
            state.final_weight=0;
            for (_,weight) in state.epsilons.iter_mut()
                .chain(state.transitions.iter_mut().flat_map(|(_,branches)|branches.iter_mut())) {
                *weight=0;
            }
            continue;
        }
        profile.live_states+=1;
        for &(next,weight) in &state.epsilons {
            if weight==0{continue;}
            let dest=&mut contexts[next as usize*words..(next as usize+1)*words];
            if has(&source,root){insert(dest,root,root);}
            else if !has(dest,root){
                for (x,y) in dest.iter_mut().zip(&source){*x|=*y;}
                profile.propagation_words+=words;
            }
            profile.live_edges_after+=1;
        }
        for (label,branches) in &mut state.transitions {
            let possible=*label==DEFAULT_LABEL || source.iter()
                .zip(&domain.allowed[*label as usize*words..(*label as usize+1)*words])
                .any(|(x,y)|x&y!=0);
            if !possible {
                profile.blocked_labels+=1;
                for (_,weight) in branches.iter_mut(){*weight=0;}
                // Even an entirely empty row remains present as a guard.
                continue;
            }
            let context=if *label==DEFAULT_LABEL{root}else{domain.target[*label as usize] as usize};
            for &(next,weight) in branches.iter(){
                if weight==0{continue;}
                insert(&mut contexts[next as usize*words..(next as usize+1)*words],context,root);
                profile.live_edges_after+=1;
                profile.forgotten_defaults+=usize::from(*label==DEFAULT_LABEL);
            }
        }
    }
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
}
