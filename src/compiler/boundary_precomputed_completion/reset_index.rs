//! Component-only exact reset observations over factors of the FULL model
//! vocabulary. No caller, partner, boundary word list, or initial seed set is
//! an input. Missing transitions decline; a recorded dead edge means empty.
use glrmask_lexer::__private::automata::lexer::{Lexer,tokenizer::Tokenizer};
use rustc_hash::FxHashMap;
use std::{collections::BTreeMap,sync::Arc};
const DEAD:u32=u32::MAX;
#[derive(Debug)]
struct Node{frontier:Vec<u32>,matched:Vec<u32>,future:Vec<u32>,edges:BTreeMap<u8,u32>}
#[derive(Debug,Default)]
pub struct Stats{pub edges:usize,pub frontier_cells:usize,pub labels:usize,pub scanned_states:usize,pub factor_byte_visits:usize}
/// A proof under construction or decoded from the legacy wire. Its raw
/// subsets are witnesses, not part of the executable reset relation.
pub(crate) struct ResetCertificate {
    source: Arc<Tokenizer>,
    nodes: Vec<Node>,
}

/// The certified observations are a compact partial byte automaton. A
/// missing edge declines the acceleration; a DEAD edge proves emptiness.
pub(crate) struct ResetIndex {
    source: Arc<Tokenizer>,
    rows: Box<[ResetRow]>,
    labels: Box<[u32]>,
    edges: Box<[(u8, u32)]>,
}

struct ResetRow {
    matched: u32,
    future: u32,
    edges: u32,
}
fn outputs(source:&Tokenizer,frontier:&[u32])->(Vec<u32>,Vec<u32>){
    let mut matched=frontier.iter().flat_map(|&q|source.matched_terminals_iter(q)).collect::<Vec<_>>();matched.sort_unstable();matched.dedup();
    let mut future=frontier.iter().flat_map(|&q|source.possible_future_terminals_iter(q)).collect::<Vec<_>>();future.sort_unstable();future.dedup();(matched,future)
}
+
    pub fn prepare(source:Arc<Tokenizer>,vocabulary:&[Vec<u8>])->Option<Self>{
        if source.num_states()==0||source.num_states()>500_000||source.has_virtual_residual_runtime()||vocabulary.iter().any(|w|w.len()>4096){return None}
        if vocabulary.iter().try_fold(0usize,|n,w|n.checked_add(w.len()))?>4_000_000{return None}
        let mut root=source.deterministic_reset_states().iter().flat_map(|&q|source.singleton_epsilon_closure(q).into_vec()).collect::<Vec<_>>();root.sort_unstable();root.dedup();
        let(matched,future)=outputs(source.as_ref(),&root);let mut stats=Stats{frontier_cells:root.len(),labels:matched.len()+future.len(),..Default::default()};
        let mut ids=FxHashMap::default();ids.insert(root.clone(),0);
        let mut nodes=vec![Node{frontier:root,matched,future,edges:BTreeMap::new()}];
        for word in vocabulary{for cut in 0..word.len(){let mut q=0usize;for &b in &word[cut..]{
            stats.factor_byte_visits+=1;if stats.factor_byte_visits>32_000_000{return None}
            let target=if let Some(&target)=nodes[q].edges.get(&b){target}else{
                stats.scanned_states=stats.scanned_states.checked_add(nodes[q].frontier.len())?;if stats.scanned_states>64_000_000{return None}
                let mut next=source.step_all(&nodes[q].frontier,b).into_vec();next.sort_unstable();next.dedup();
                let target=if next.is_empty(){DEAD}else if let Some(&id)=ids.get(&next){id}else{
                    if nodes.len()>=100_000{return None}
                    stats.frontier_cells+=next.len();if stats.frontier_cells>2_000_000{return None}
                    let(matched,future)=outputs(source.as_ref(),&next);stats.labels+=matched.len()+future.len();if stats.labels>8_000_000{return None}
                    let id=nodes.len()as u32;ids.insert(next.clone(),id);nodes.push(Node{frontier:next,matched,future,edges:BTreeMap::new()});id
                };
                stats.edges+=1;if stats.edges>1_000_000{return None}nodes[q].edges.insert(b,target);target
            };
            if target==DEAD{break}q=target as usize;
        }}}
        Some(Self { source, nodes })
    }
    /// Exhaustive independent certificate check: every observation and every
    /// recorded transition is checked against the immutable physical source.
    fn verify(&self)->bool{
        let mut root=self.source.deterministic_reset_states().iter().flat_map(|&q|self.source.singleton_epsilon_closure(q).into_vec()).collect::<Vec<_>>();root.sort_unstable();root.dedup();
        if self.nodes[0].frontier!=root{return false}
        for node in &self.nodes{
            if !node.frontier.windows(2).all(|x|x[0]<x[1]){return false}
            let(m,f)=outputs(self.source.as_ref(),&node.frontier);if node.matched!=m||node.future!=f{return false}
            for(&b,&target)in &node.edges{let mut next=self.source.step_all(&node.frontier,b).into_vec();next.sort_unstable();next.dedup();
                if target==DEAD{if !next.is_empty(){return false}}else if self.nodes.get(target as usize).is_none_or(|n|n.frontier!=next){return false}
            }
        }true
    }
    pub(crate) fn certify(self) -> Option<ResetIndex> {
        if !self.verify() { return None; }
        let mut rows = Vec::with_capacity(self.nodes.len() + 1);
        let mut labels = Vec::new();
        let mut edges = Vec::new();
        for node in self.nodes {
            let matched = u32::try_from(labels.len()).ok()?;
            labels.extend(node.matched);
            let future = u32::try_from(labels.len()).ok()?;
            labels.extend(node.future);
            let edge = u32::try_from(edges.len()).ok()?;
            edges.extend(node.edges);
            rows.push(ResetRow { matched, future, edges: edge });
        }
        let end = u32::try_from(labels.len()).ok()?;
        rows.push(ResetRow {
            matched: end, future: end, edges: u32::try_from(edges.len()).ok()?,
        });
        Some(ResetIndex {
            source: self.source,
            rows: rows.into_boxed_slice(),
            labels: labels.into_boxed_slice(),
            edges: edges.into_boxed_slice(),
        })
    }
}

include!("reset_wire.rs");

impl ResetIndex {
    pub(crate) fn source(&self) -> &Tokenizer { &self.source }

    fn row(&self, id: u32) -> Option<(&ResetRow, &ResetRow)> {
        let id = id as usize;
        Some((self.rows.get(id)?, self.rows.get(id.checked_add(1)?)?))
    }

    pub(crate) fn step_id(&self, id: u32, byte: u8) -> Option<u32> {
        if id == DEAD { return Some(DEAD); }
        let (row, next) = self.row(id)?;
        let edges = &self.edges[row.edges as usize..next.edges as usize];
        let index = edges.binary_search_by_key(&byte, |&(b, _)| b).ok()?;
        Some(edges[index].1)
    }

    pub(crate) fn observation(&self, id: u32) -> Option<(&[u32], &[u32])> {
        if id == DEAD { return Some((&[], &[])); }
        let (row, next) = self.row(id)?;
        Some((
            &self.labels[row.matched as usize..row.future as usize],
            &self.labels[row.future as usize..next.matched as usize],
        ))
    }
}
