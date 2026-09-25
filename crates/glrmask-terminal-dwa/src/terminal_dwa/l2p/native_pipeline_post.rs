//! Exact native-arena equivalent of the existing terminal postprocess passes.
use super::*;
use crate::ds::bitset::BitSet;
use crate::automata::weighted_u32::nwa::NWAState;
type Branches=SmallVec<[(u32,u32);1]>;
#[derive(Clone,Default,PartialEq,Eq,Hash)]
struct Row { final_w:Option<u32>, eps:Branches, edges:Vec<(i32,Branches)> }
#[derive(Clone)]
struct Arena { starts:Vec<u32>, states:Vec<Row> }
#[derive(Default,Debug)]
pub struct PostProfile { pub import_ms:f64,pub follows_ms:f64,pub prune_ms:f64,pub canonical_ms:f64,pub crossing_ms:f64,pub export_ms:f64,pub input_states:usize,pub follow_states:usize,pub pruned_states:usize,pub canonical_states:usize,pub output_states:usize }
fn ms(t:Instant)->f64 { t.elapsed().as_secs_f64()*1000.0 }
fn import(n:&NWA,p:&mut Pool)->Option<Arena>{
 let count=n.states().len(); if count==0||count>=16384||n.start_states().iter().any(|&s|s as usize>=count){return None}
 let mut edges=0usize;let mut states=Vec::with_capacity(count);
 for s in n.states(){
  let final_w=match &s.final_weight{Some(w)=>Some(p.import(w)?),None=>None};
  let mut eps=Branches::with_capacity(s.epsilons.len());for(t,w)in &s.epsilons{if *t as usize>=count{return None}eps.push((*t,p.import(w)?));}
  let mut out=Vec::with_capacity(s.transitions.len());for(&l,bs)in &s.transitions{if l<0{return None}let mut v=Branches::with_capacity(bs.len());for(t,w)in bs{if *t as usize>=count{return None}v.push((*t,p.import(w)?));}out.push((l,v));}
  edges+=eps.len()+out.iter().map(|(_,bs)|bs.len()).sum::<usize>();if edges>100_000||p.failed{return None}
  states.push(Row{final_w,eps,edges:out});
 }
 let a=Arena{states,starts:n.start_states().to_vec()};if topo(&a).len()!=count{return None}Some(a)
}
fn retain(a:&mut Arena,keep:&[bool],drop_empty:bool){
 if keep.iter().all(|x|*x){return}
 let mut ids=vec![u32::MAX;a.states.len()];let mut new=Vec::with_capacity(keep.iter().filter(|x|**x).count());
 for (s,row)in std::mem::take(&mut a.states).into_iter().enumerate(){if keep[s]{ids[s]=new.len()as u32;new.push(row);}}
 for row in &mut new{
  row.eps.retain(|x|keep[x.0 as usize]&&(!drop_empty||x.1!=0));for(d,_)in &mut row.eps{*d=ids[*d as usize];}
  for(_,bs)in &mut row.edges{bs.retain(|x|keep[x.0 as usize]&&(!drop_empty||x.1!=0));for(d,_)in bs{*d=ids[*d as usize];}}
  row.edges.retain(|(_,bs)|!bs.is_empty());
 }
 a.starts.retain(|&s|keep[s as usize]);for s in &mut a.starts{*s=ids[*s as usize]}a.states=new;
}
fn reachable(a:&mut Arena){
 let mut seen=vec![false;a.states.len()];let mut queue=a.starts.clone();let mut i=0;
 for&s in &queue{seen[s as usize]=true}
 while i<queue.len(){let row=&a.states[queue[i]as usize];for&(d,_)in row.eps.iter().chain(row.edges.iter().flat_map(|(_,bs)|bs)){if !std::mem::replace(&mut seen[d as usize],true){queue.push(d)}}i+=1;}
 retain(a,&seen,false);
}
fn coreachable(a:&mut Arena){
 let n=a.states.len();let mut back=vec![Vec::new();n];let mut keep=vec![false;n];let mut queue=Vec::new();
 for(s,row)in a.states.iter().enumerate(){if row.final_w.is_some_and(|w|w!=0){keep[s]=true;queue.push(s)}for&(d,w)in row.eps.iter().chain(row.edges.iter().flat_map(|(_,bs)|bs)){if w!=0{back[d as usize].push(s);}}}
 let mut i=0;while i<queue.len(){for &pred in &back[queue[i]]{if !std::mem::replace(&mut keep[pred],true){queue.push(pred)}}i+=1;}
 retain(a,&keep,true);
}
fn topo(a:&Arena)->Vec<usize>{
 let mut indeg=vec![0usize;a.states.len()];for row in &a.states{for&(d,_)in row.eps.iter().chain(row.edges.iter().flat_map(|(_,bs)|bs)){indeg[d as usize]+=1;}}
 let mut queue=indeg.iter().enumerate().filter_map(|(s,&n)|(n==0).then_some(s)).collect::<Vec<_>>();let mut i=0;
 while i<queue.len(){for&(d,_)in a.states[queue[i]].eps.iter().chain(a.states[queue[i]].edges.iter().flat_map(|(_,bs)|bs)){indeg[d as usize]-=1;if indeg[d as usize]==0{queue.push(d as usize)}}i+=1;}queue
}
fn follow_product(a:Arena,rows:&[Option<&BitSet>],ignore:Option<u32>,transparent:Option<&BitSet>)->Option<Arena>{
 if rows.iter().all(Option::is_none){return Some(a)}
 let mut states=Vec::<Row>::new();let mut payload=Vec::<(u32,u32)>::new();let mut ids=FxHashMap::default();
 let intern=|key:(u32,u32),ids:&mut FxHashMap<(u32,u32),u32>,states:&mut Vec<Row>,payload:&mut Vec<(u32,u32)>|->u32{
  if let Some(&id)=ids.get(&key){return id}let id=states.len()as u32;states.push(Row::default());payload.push(key);ids.insert(key,id);id
 };
 let starts=a.starts.iter().map(|&s|intern((s,u32::MAX),&mut ids,&mut states,&mut payload)).collect();let mut i=0;let mut edge_count=0;
 while i<payload.len(){if payload.len()>MAX_STATES||edge_count>1_000_000{return None}let (source,prev)=payload[i];let original=&a.states[source as usize];let mut row=Row{final_w:original.final_w,..Default::default()};
  for&(d,w)in &original.eps{row.eps.push((intern((d,prev),&mut ids,&mut states,&mut payload),w));edge_count+=1;}
  for&(l,ref bs)in &original.edges{
   let trans=ignore==Some(l as u32)||transparent.is_some_and(|t|t.contains(l as usize));
   let next_prev=if trans{prev}else if (l as usize)<rows.len(){if rows.get(prev as usize).and_then(|x|*x).is_some_and(|r|r.contains(l as usize)){continue}l as u32}else{u32::MAX};
   let out=bs.iter().map(|&(d,w)|(intern((d,next_prev),&mut ids,&mut states,&mut payload),w)).collect::<Branches>();edge_count+=out.len();if !out.is_empty(){row.edges.push((l,out));}
  }
  states[i]=row;i+=1;
 }
 if states.len()>MAX_STATES||edge_count>1_000_000{return None}
 Some(Arena{starts,states})
}
fn canonicalize(a:&mut Arena,p:&mut Pool)->Option<()>{
 if a.states.len()<=1{return Some(())}reachable(a);let order=topo(a);if order.len()!=a.states.len(){return None}
 let mut mapped=vec![u32::MAX;a.states.len()];let mut states=Vec::<Row>::with_capacity(a.states.len());let mut seen=FxHashMap::<Row,u32>::default();let mut merged=0;
 let mut remap_branches=|bs:&[(u32,u32)],mapped:&[u32]|->Branches{
  let mut pairs=bs.iter().map(|&(d,w)|(mapped[d as usize],w)).collect::<Vec<_>>();pairs.sort_by_key(|x|x.0);let mut out=Branches::with_capacity(pairs.len());
  for(d,w)in pairs{if let Some((last,weight))=out.last_mut(){if *last==d{*weight=p.join(*weight,w);continue}}out.push((d,w));}out
 };
 for s in order.into_iter().rev(){let original=&a.states[s];let eps=remap_branches(&original.eps,&mapped);let edges=original.edges.iter().filter_map(|(l,bs)|{let v=remap_branches(bs,&mapped);(!v.is_empty()).then_some((*l,v))}).collect();let row=Row{final_w:original.final_w,eps,edges};
  let id=if let Some(&id)=seen.get(&row){merged+=1;id}else{let id=states.len()as u32;seen.insert(row.clone(),id);states.push(row);id};mapped[s]=id;
 }
 if p.failed{return None}if merged!=0{let mut starts=Vec::new();for&s in &a.starts{let id=mapped[s as usize];if !starts.contains(&id){starts.push(id)}}a.starts=starts;a.states=states;}Some(())
}
fn crossing(a:Arena,owner:&[u32],start_owner:u32)->Option<Arena>{
 if a.states.len()>MAX_STATES{return None}let mut ids=vec![u32::MAX;a.states.len()*2];let mut payload=Vec::<(u32,bool)>::new();let mut states=Vec::<Row>::new();
 let intern=|d:u32,seen:bool,ids:&mut Vec<u32>,states:&mut Vec<Row>,payload:&mut Vec<(u32,bool)>|{let slot=d as usize*2+usize::from(seen);if ids[slot]!=u32::MAX{return ids[slot]}let id=states.len()as u32;ids[slot]=id;states.push(Row::default());payload.push((d,seen));id};
 let starts=a.starts.iter().map(|&s|intern(s,false,&mut ids,&mut states,&mut payload)).collect();let mut i=0;let mut edge_count=0;
 while i<payload.len(){if states.len()>MAX_STATES||edge_count>1_000_000{return None}let(s,seen)=payload[i];let orig=&a.states[s as usize];let mut row=Row{final_w:if seen{orig.final_w}else{None},..Default::default()};
  for&(d,w)in &orig.eps{row.eps.push((intern(d,seen,&mut ids,&mut states,&mut payload),w));edge_count+=1;}
  for&(l,ref bs)in &orig.edges{let crossed=seen||*owner.get(l as usize)?!=start_owner;let out=bs.iter().map(|&(d,w)|(intern(d,crossed,&mut ids,&mut states,&mut payload),w)).collect::<Branches>();edge_count+=out.len();if !out.is_empty(){row.edges.push((l,out));}}
  states[i]=row;i+=1;
 }
 if states.len()>MAX_STATES||edge_count>1_000_000{return None}
 Some(Arena{starts,states})
}
fn export(a:Arena,p:&mut Pool)->NWA{NWA::from_parts(a.states.into_iter().map(|s|NWAState{final_weight:s.final_w.map(|w|p.export(w)),epsilons:s.eps.into_iter().map(|(d,w)|(d,p.export(w))).collect(),transitions:s.edges.into_iter().map(|(l,bs)|(l,bs.into_iter().map(|(d,w)|(d,p.export(w))).collect())).collect()}).collect(),a.starts)}
fn prepare(n:&NWA,follows:&BTreeMap<u32,BitSet>,num_terminals:usize,ignore:Option<u32>,transparent:Option<&BitSet>,owner:&[u32],start_owner:u32)->Option<(Arena,Pool,PostProfile)>{
 if num_terminals>65536||owner.len()!=num_terminals{return None}
 let mut p=Pool::new();let t=Instant::now();let a=import(n,&mut p)?;let stats=PostProfile{input_states:a.states.len(),import_ms:ms(t),..Default::default()};

 prepare_arena(a,p,stats,follows,num_terminals,ignore,transparent,owner,start_owner)
}
fn prepare_arena(a:Arena,mut p:Pool,mut stats:PostProfile,follows:&BTreeMap<u32,BitSet>,num_terminals:usize,ignore:Option<u32>,transparent:Option<&BitSet>,owner:&[u32],start_owner:u32)->Option<(Arena,Pool,PostProfile)>{
 if num_terminals>65536||owner.len()!=num_terminals{return None}
 let t=Instant::now();let rows=(0..num_terminals).map(|t|follows.get(&(t as u32)).filter(|b|!b.is_zero())).collect::<Vec<_>>();let mut a=follow_product(a,&rows,ignore,transparent)?;stats.follows_ms=ms(t);stats.follow_states=a.states.len();
 let t=Instant::now();coreachable(&mut a);stats.prune_ms=ms(t);stats.pruned_states=a.states.len();
 let t=Instant::now();canonicalize(&mut a,&mut p)?;stats.canonical_ms=ms(t);stats.canonical_states=a.states.len();
 let t=Instant::now();let mut a=crossing(a,owner,start_owner)?;coreachable(&mut a);stats.crossing_ms=ms(t);stats.output_states=a.states.len();
 if p.failed{return None}Some((a,p,stats))
}

pub fn prepare_export(n:&NWA,follows:&BTreeMap<u32,BitSet>,num_terminals:usize,ignore:Option<u32>,transparent:Option<&BitSet>,owner:&[u32],start_owner:u32)->Option<(NWA,PostProfile)>{
 let(a,mut p,mut stats)=prepare(n,follows,num_terminals,ignore,transparent,owner,start_owner)?;let t=Instant::now();let out=export(a,&mut p);stats.export_ms=ms(t);Some((out,stats))
}
pub fn compile(n:&NWA,follows:&BTreeMap<u32,BitSet>,num_terminals:usize,ignore:Option<u32>,transparent:Option<&BitSet>,owner:&[u32],start_owner:u32)->Option<(DWA,PostProfile,Profile)>{
 if !policy_supported(){return None}
 let(a,p,stats)=prepare(n,follows,num_terminals,ignore,transparent,owner,start_owner)?;

 finish_arena(a,p,stats)
}
fn finish_arena(a:Arena,p:Pool,stats:PostProfile)->Option<(DWA,PostProfile,Profile)>{
 if !policy_supported(){return None}
 if a.states.is_empty()||a.starts.is_empty()||a.states.len()>=16384{return None}
 for s in &a.states{for (_,bs)in &s.edges{for&(d,_)in bs{if !a.states[d as usize].eps.is_empty(){return None}}}}
 let t=Instant::now();let mut states=Vec::with_capacity(a.states.len());
 for row in a.states{
  let mut keys=FxHashMap::<u32,usize>::default();let mut groups=Vec::<SourceGroup>::new();
  for(l,bs)in row.edges{let direct=bs.len()==1;for(d,w)in bs{let group=if let Some(&g)=keys.get(&w){g}else{let g=groups.len();groups.push(SourceGroup{weight:w,edges:Vec::new()});keys.insert(w,g);g};groups[group].edges.push((l,d,direct));}}
  states.push(State{final_w:row.final_w.unwrap_or(0),groups,eps:row.eps.into_vec()});
 }
 let(dwa,core)=run_imported(states,&a.starts,p,true,ms(t))?;Some((dwa,stats,core))
}

#[cfg(test)] mod tests {
use super::*;
use crate::terminal_dwa::{l2p::postprocess as original,scope::{BoundaryOwnership,ImmediateComponentId}};
fn reference(n:&NWA,rows:&BTreeMap<u32,BitSet>,count:usize,ignore:Option<u32>,transparent:Option<&BitSet>,owner:&[u32],start:u32)->NWA{
 let mut offsets=Vec::new();let mut os=Vec::new();for(t,&o)in owner.iter().enumerate(){if t==0||owner[t-1]!=o{offsets.push(t as u32);os.push(ImmediateComponentId(o));}}
 let ownership=BoundaryOwnership::from_leaf_layout(&offsets,count as u32,&os,owner.iter().max().unwrap()+1).unwrap();
 let mut result=n.clone();original::apply_disallowed_follow_constraints(&mut result,rows,count,ignore,transparent);original::prune_non_coreachable_states(&mut result);original::canonicalize_acyclic_nwa(&mut result);result=original::filter_nwa_to_crossing_paths(&result,&ownership,ImmediateComponentId(start));original::prune_non_coreachable_states(&mut result);result
}
fn next(seed:&mut u64)->usize{*seed=seed.wrapping_mul(6364136223846793005).wrapping_add(1);(*seed>>32)as usize}
fn w(seed:&mut u64)->Weight{
 match next(seed)%9{0=>Weight::empty(),1=>Weight::all(),_=>Weight::from_per_tsid_token_sets([0u32,3,4,u32::MAX].into_iter().map(|row|(row,[0u32,1,63,64,255,256,511].into_iter().filter(|_|next(seed)%2==0).collect()))) }
}
#[test]fn native_post_matches_every_original_row_on_generated_dags(){
 let mut rng=723419u64;let mut fused_eligible=0;
 for case in 0..512{
  let n=3+next(&mut rng)%23;let count=7;let mut nwa=NWA::new(0,0);for _ in 0..n{nwa.add_state();}
  nwa.set_start_states(if case%19==0{vec![]}else if case%13==0{vec![0,1,0]}else{vec![0]});
  for s in 0..n{
   if next(&mut rng)%3!=0{nwa.set_final_weight(s as u32,w(&mut rng));}
   for d in s+1..n{if next(&mut rng)%7==0{nwa.add_epsilon(s as u32,d as u32,w(&mut rng));}
    if next(&mut rng)%4==0{let l=next(&mut rng)%count;let weight=w(&mut rng);nwa.add_transition(s as u32,l as i32,d as u32,weight.clone());if case%11==0{nwa.add_transition(s as u32,l as i32,d as u32,weight);}}
   }
   if case%17==0{nwa.states_mut()[s].transitions.entry(6).or_default();}
  }
  let mut rows=BTreeMap::new();if case%23!=0{for t in 0..count{let mut row=BitSet::new(count);for l in 0..count{if next(&mut rng)%3==0{row.set(l)}}rows.insert(t as u32,row);}}
  let mut transparent=BitSet::new(count);if case%2==0{transparent.set(5);}let transparent=(case%5!=0).then_some(&transparent);let ignore=(case%3==0).then_some(1u32);
  let owner=(0..count).map(|l|((l+case)%3)as u32).collect::<Vec<_>>();let start=(case%3)as u32;
  let before=(nwa.start_states().to_vec(),nwa.states().to_vec());let baseline=reference(&nwa,&rows,count,ignore,transparent,&owner,start);
  let(candidate,_)=prepare_export(&nwa,&rows,count,ignore,transparent,&owner,start).expect("bounded supported input");
  assert_eq!(baseline.start_states(),candidate.start_states(),"starts case={case}");assert_eq!(baseline.states(),candidate.states(),"every row case={case}");
  if let Some((a,_,_))=compile(&nwa,&rows,count,ignore,transparent,&owner,start){
   let b=super::super::compile_sparse(&baseline).expect("same native eligibility").0;
   assert!(super::super::graph_isomorphism(&a,&b).is_ok(),"fused case={case}");fused_eligible+=1;
  }
  assert_eq!(before,(nwa.start_states().to_vec(),nwa.states().to_vec()),"mutation case={case}");
 }
 eprintln!("NATIVE_POST 512 whole-array comparisons, fused_eligible={fused_eligible}");
}
#[test]fn native_post_declines_unsupported_without_mutation(){
 let rows=BTreeMap::new();let mut n=NWA::new(0,0);n.add_state();n.set_start_states(vec![0]);n.add_transition(0,0,0,Weight::all());assert!(prepare_export(&n,&rows,2,None,None,&[0,1],0).is_none());
 let mut n=NWA::new(0,0);n.add_state();n.add_state();n.set_start_states(vec![0]);n.add_transition(0,0,1,Weight::from_token_set_for_tsid(0,[512].into_iter().collect()));assert!(prepare_export(&n,&rows,2,None,None,&[0,1],0).is_none());
 let mut n=NWA::new(0,0);n.add_state();n.set_start_states(vec![99]);assert!(prepare_export(&n,&rows,2,None,None,&[0,1],0).is_none());
 let mut n=NWA::new(0,0);n.add_state();n.set_start_states(vec![0]);n.add_transition(0,-1,0,Weight::all());assert!(prepare_export(&n,&rows,2,None,None,&[0,1],0).is_none());
 assert!(prepare_export(&n,&rows,2,None,None,&[0],0).is_none());
}

}


/// Owns the exact seeded graph and finite coefficients throughout lexical
/// event emission, postprocessing, determinization, and minimization.
pub struct NativeSink { arena:Arena, pool:Pool, num_tsids:u32, edges:usize }
impl NativeSink {
 pub fn from_seed(seed:&NWA,num_tsids:u32)->Option<Self> {
  if num_tsids==0||!policy_supported(){return None}
  let mut pool=Pool::new();let arena=import(seed,&mut pool)?;
  // With one ALL-final leaf and exclusively finite coefficients, the old
  // collapse_always_allowed pass is provably an identity. Do not assume that
  // contract for arbitrary graphs.
  let mut all_finals=0usize;let mut edges=0usize;
  for row in &arena.states {
   if row.final_w==Some(1) {all_finals+=1;if !row.edges.is_empty()||!row.eps.is_empty(){return None}}
   for &(_,w) in row.eps.iter().chain(row.edges.iter().flat_map(|(_,bs)|bs)) {if w==1{return None}edges+=1;}
  }
  if all_finals!=1{return None}
  Some(Self{arena,pool,num_tsids,edges})
 }
 pub fn states_len(&self)->usize {self.arena.states.len()}
 pub fn add_state(&mut self)->u32 {
  if self.arena.states.len()>=16384 {self.pool.failed=true;return 0}
  let id=self.arena.states.len() as u32;self.arena.states.push(Row::default());id
 }
 fn coefficient(&mut self,end:u32,valid:bool,bits:&[u64;8])->u32 {
  if !valid{self.pool.failed=true;return 0}if bits.iter().all(|&b|b==0){return 0}if end!=self.num_tsids-1 {self.pool.failed=true;return 0}
  let b=self.pool.intern_bits(*bits);
  if b==0 {0}else{self.pool.intern(smallvec![Run{lo:0,hi:end,b}])}
 }
 pub fn append_epsilon(&mut self,from:u32,to:u32,end:u32,valid:bool,bits:&[u64;8]) {
  if self.edges>=100_000||from as usize>=self.arena.states.len()||to as usize>=self.arena.states.len(){self.pool.failed=true;return}
  let w=self.coefficient(end,valid,bits);self.arena.states[from as usize].eps.push((to,w));self.edges+=1;
 }
 pub fn append_transition(&mut self,from:u32,label:i32,to:u32,end:u32,valid:bool,bits:&[u64;8]) {
  if self.edges>=100_000||label<0||from as usize>=self.arena.states.len()||to as usize>=self.arena.states.len(){self.pool.failed=true;return}
  let w=self.coefficient(end,valid,bits);let edges=&mut self.arena.states[from as usize].edges;
  match edges.binary_search_by_key(&label,|x|x.0) {
   Ok(i)=>edges[i].1.push((to,w)),Err(i)=>edges.insert(i,(label,smallvec![(to,w)])),
  }
  self.edges+=1;
 }
 pub fn export_raw(&mut self)->Option<NWA> {if self.pool.failed{return None}Some(export(self.arena.clone(),&mut self.pool))}
 pub fn finish(self,follows:&BTreeMap<u32,BitSet>,num_terminals:usize,ignore:Option<u32>,transparent:Option<&BitSet>,owner:&[u32],start_owner:u32)->Option<(DWA,PostProfile,Profile)> {
  if self.pool.failed{return None}
  let stats=PostProfile{input_states:self.arena.states.len(),..Default::default()};
  let(a,p,stats)=prepare_arena(self.arena,self.pool,stats,follows,num_terminals,ignore,transparent,owner,start_owner)?;
  // Keep the pre-existing specialized depth-two constructor unchanged.
  let mut depth=vec![None::<usize>;a.states.len()];
  for s in topo(&a).into_iter().rev() {
   let mut d=a.states[s].final_w.filter(|&w|w!=0).map(|_|0usize);
   for &(t,_) in &a.states[s].eps {if let Some(x)=depth[t as usize]{d=Some(d.unwrap_or(0).max(x));}}
   for &(_,ref bs) in &a.states[s].edges {for &(t,_) in bs {if let Some(x)=depth[t as usize]{d=Some(d.unwrap_or(0).max(x+1));}}}
   depth[s]=d;
  }
  if a.starts.iter().filter_map(|&s|depth[s as usize]).max().is_none_or(|d|d<=2){return None}
  finish_arena(a,p,stats)
 }
}
