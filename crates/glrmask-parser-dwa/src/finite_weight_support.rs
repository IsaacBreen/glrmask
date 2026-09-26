//! Optional positive-NWA coordinate support. This keeps every state, exact
//! stack label, target and empty guard key. It is a weighted-language pruning
//! theorem only; contextual parser normalization still requires its stronger
//! final prefix/domain oracle before this experiment can be accepted.
use super::*;
#[derive(Debug,Default)]pub(super)struct Stats{pub changed:usize,pub emptied:usize,pub before_weights:usize,pub after_weights:usize,pub unique_live_weights:usize}
pub(super)fn restrict(states:&mut[FastBoundaryNwaState],starts:&[u32],pool:&mut FastBoundaryWeightInterner,forward:bool,backward:bool)->Option<Stats>{
 let n=states.len();if starts.iter().any(|q|*q as usize>=n){return None;}
 if states.iter().any(|s|s.transitions.iter().any(|(l,b)|is_negative_label(*l)||b.iter().any(|(q,w)|*q as usize>=n||*w as usize>=pool.values.len()))||s.epsilons.iter().any(|(q,w)|*q as usize>=n||*w as usize>=pool.values.len())||s.final_weight as usize>=pool.values.len()){return None;}
 let topo=fast_boundary_topological_order(states)?;
 let mut f=vec![if forward{0}else{1};n];let mut r=vec![if backward{0}else{1};n];
 let mut stats=Stats{before_weights:pool.values.len(),..Default::default()};
 if forward{for &q in starts{f[q as usize]=1;}for &q in &topo{let row=&states[q as usize];for &(target,w)in row.epsilons.iter().chain(row.transitions.iter().flat_map(|(_,b)|b.iter())){let add=pool.intersection(f[q as usize],w);f[target as usize]=pool.union(f[target as usize],add);}}}
 if backward{for &q in topo.iter().rev(){let row=&states[q as usize];let mut value=row.final_weight;for &(target,w)in row.epsilons.iter().chain(row.transitions.iter().flat_map(|(_,b)|b.iter())){let add=pool.intersection(r[target as usize],w);value=pool.union(value,add);}r[q as usize]=value;}}
 if pool.failed{return None;}
 for(q,row)in states.iter_mut().enumerate(){let f=f[q];row.final_weight=pool.intersection(row.final_weight,f);
  for(target,w)in row.epsilons.iter_mut().chain(row.transitions.iter_mut().flat_map(|(_,b)|b.iter_mut())){let a=pool.intersection(*w,f);let a=pool.intersection(a,r[*target as usize]);stats.changed+=usize::from(a!=*w);stats.emptied+=usize::from(a==0&&*w!=0);*w=a;}
 }
 if pool.failed{return None;}
 let mut live=std::collections::BTreeSet::new();for row in states{live.insert(row.final_weight);live.extend(row.epsilons.iter().chain(row.transitions.iter().flat_map(|(_,b)|b.iter())).map(|(_,w)|*w));}
 stats.after_weights=pool.values.len();stats.unique_live_weights=live.len();Some(stats)
}
#[cfg(test)]mod tests{
 use super::*;
 fn generic(states:&[FastBoundaryNwaState],pool:&FastBoundaryWeightInterner)->NWA{
  let mut nwa=NWA::new(0,0);for _ in states{nwa.add_state();}nwa.set_start_states(vec![0]);
  for(q,row)in states.iter().enumerate(){if row.final_weight!=0{nwa.set_final_weight(q as u32,pool.to_weight(row.final_weight));}for &(t,w)in &row.epsilons{nwa.add_epsilon(q as u32,t,pool.to_weight(w));}for &(label,ref branches)in &row.transitions{nwa.states_mut()[q].transitions.insert(label,branches.iter().map(|&(t,w)|(t,pool.to_weight(w))).collect());}}
  nwa
 }
 #[test]fn positive_weight_support_matches_independent_weighted_words_and_keeps_guards(){
  let mut seed=55u64;let mut next=||{seed=seed.wrapping_mul(6364136223846793005).wrapping_add(1);(seed>>32)as usize};let mut changes=0;
  for _ in 0..256{let n=4+next()%7;let mut pool=FastBoundaryWeightInterner::new(1,64).unwrap();let ws=(0..16u64).map(|x|pool.intern(smallvec::smallvec![x])).collect::<Vec<_>>();
   let mut source=Vec::new();for q in 0..n{let mut row=FastBoundaryNwaState{epsilons:Vec::new(),transitions:Vec::new(),final_weight:ws[next()%ws.len()]};for label in [0,1,2,DEFAULT_LABEL]{let mut bs=SmallVec::new();for t in q+1..n{if next()%4==0{bs.push((t as u32,ws[next()%ws.len()]));}}row.transitions.push((label,bs));}for t in q+1..n{if next()%4==0{row.epsilons.push((t as u32,ws[next()%ws.len()]));}}source.push(row);}
   let left=crate::automata::weighted::determinize::determinize(&generic(&source,&pool)).unwrap();
   for(f,b)in[(true,false),(false,true),(true,true)]{let mut candidate=source.clone();let s=restrict(&mut candidate,&[0],&mut pool,f,b).unwrap();changes+=s.changed;
    for(a,b)in source.iter().zip(&candidate){assert_eq!(a.transitions.iter().map(|(l,b)|(*l,b.iter().map(|x|x.0).collect::<Vec<_>>())).collect::<Vec<_>>(),b.transitions.iter().map(|(l,b)|(*l,b.iter().map(|x|x.0).collect::<Vec<_>>())).collect::<Vec<_>>());}
    let right=crate::automata::weighted::determinize::determinize(&generic(&candidate,&pool)).unwrap();
    assert!(crate::automata::weighted::equivalence::find_difference(&left,&right).unwrap().is_none());assert!(crate::automata::weighted::equivalence::find_difference(&right,&left).unwrap().is_none());
   }
  }assert!(changes>0);
 }
}
