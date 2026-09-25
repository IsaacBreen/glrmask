//! Necessary signed-path support with exact finite mask/top correlation.
//!
//! U represents an unknown stack top for its mask coordinates. K[a] carries
//! coordinates whose top is known to be a. READ(a) selects U|K[a], then forgets
//! the exposed tail; PUSH(a) makes that top exact. This deliberately does not
//! assume an LR predecessor relation for stacks written inside a template.
//! Every propagated coordinate overapproximates a concrete execution by
//! induction over the original DAG. Removing unsupported edge coordinates
//! therefore preserves the complete input/output-stack relation. This alone
//! does NOT certify contextual DEFAULT normalization; callers must validate it.
use super::*;

type Domain = u32;
type Mask = FastBoundaryWeightId;

#[derive(Debug, Default)]
pub(super) struct Profile {
    pub states: usize,
    pub live_states: usize,
    pub edges_before: usize,
    pub edges_after: usize,
    pub changed_edges: usize,
    pub domains: usize,
    pub stored_pairs: usize,
    pub work: usize,
}

#[derive(Clone)]
struct Row { unknown: Mask, known: Vec<(u32, Mask)>, total: Mask }

struct Domains {
    rows: Vec<Row>,
    ids: FxHashMap<(Mask, Vec<(u32, Mask)>), Domain>,
    unions: FxHashMap<(Domain, Domain), Domain>,
    meets: FxHashMap<(Domain, Mask), Domain>,
    pairs: usize,
    work: usize,
    max_rows: usize,
}

impl Drop for Domains {
    fn drop(&mut self) {
        if compile_profile_enabled() {
            eprintln!("[glrmask/profile][weighted_top_domain_storage] rows={} max_rows={} stored_pairs={} work={} unions={} meets={}",
                self.rows.len(),self.max_rows,self.pairs,self.work,self.unions.len(),self.meets.len());
        }
    }
}

impl Domains {
    fn new() -> Self {
        Self { rows: vec![Row { unknown: 0, known: Vec::new(), total: 0 },
                          Row { unknown: 1, known: Vec::new(), total: 1 }],
            ids: FxHashMap::from_iter([((0,Vec::new()),0),((1,Vec::new()),1)]),
            unions: FxHashMap::default(), meets: FxHashMap::default(),
            pairs: 0, work: 0, max_rows: 200_000 }
    }
    fn charge(&mut self, n: usize) -> Option<()> {
        self.work = self.work.checked_add(n)?;
        (self.work <= 16_000_000).then_some(())
    }
    fn intern(&mut self, unknown: Mask, mut known: Vec<(u32,Mask)>,
              masks: &mut FastBoundaryWeightInterner) -> Option<Domain> {
        if unknown == 1 { return Some(1); }
        self.charge(known.len()+1)?;
        // Remove only the redundant uniform part, never a top distinction.
        for (_,w) in &mut known { *w = masks.difference(*w,unknown); }
        known.retain(|&(_,w)| w != 0);
        if unknown == 0 && known.is_empty() { return Some(0); }
        let key=(unknown,known);
        if let Some(&id)=self.ids.get(&key) { return Some(id); }
        if self.rows.len() >= self.max_rows
            || self.pairs.checked_add(key.1.len())? > 1_000_000 { return None; }
        let mut total=unknown;
        for &(_,w) in &key.1 { total=masks.union(total,w); }
        if !masks.allow_work(key.1.len()+1,0,0) { return None; }
        let id=self.rows.len() as Domain;
        self.pairs+=key.1.len();
        self.rows.push(Row { unknown,known:key.1.clone(),total });
        self.ids.insert(key,id);Some(id)
    }
    fn unknown(&mut self,w:Mask,masks:&mut FastBoundaryWeightInterner)->Option<Domain> {
        self.intern(w,Vec::new(),masks)
    }
    fn singleton(&mut self,a:u32,w:Mask,masks:&mut FastBoundaryWeightInterner)->Option<Domain> {
        if w==0 {Some(0)} else {self.intern(0,vec![(a,w)],masks)}
    }
    fn read(&self,id:Domain,a:u32,masks:&mut FastBoundaryWeightInterner)->Mask {
        let row=&self.rows[id as usize];
        match row.known.binary_search_by_key(&a,|&(k,_)|k) {
            Ok(index)=>masks.union(row.unknown,row.known[index].1),
            Err(_)=>row.unknown,
        }
    }
    fn meet(&mut self,id:Domain,w:Mask,masks:&mut FastBoundaryWeightInterner)->Option<Domain> {
        if id==0 || w==0 { return Some(0); }
        if w==1 { return Some(id); }
        if let Some(&v)=self.meets.get(&(id,w)) {return Some(v);}
        let row=&self.rows[id as usize];
        let u=masks.intersection(row.unknown,w);
        let known=row.known.iter().map(|&(a,x)|(a,masks.intersection(x,w))).collect();
        let result=self.intern(u,known,masks)?;
        if self.meets.len()<262_144 {self.meets.insert((id,w),result);}
        Some(result)
    }
    fn union(&mut self,a:Domain,b:Domain,masks:&mut FastBoundaryWeightInterner)->Option<Domain> {
        if a==b || b==0 || a==1 {return Some(a);}
        if a==0 || b==1 {return Some(b);}
        let key=if a<b {(a,b)} else {(b,a)};
        if let Some(&v)=self.unions.get(&key) {return Some(v);}
        let left=&self.rows[a as usize];let right=&self.rows[b as usize];
        let unknown=masks.union(left.unknown,right.unknown);
        if unknown==1 {return Some(1);}
        let count=left.known.len().checked_add(right.known.len())?;
        if count>1_000_000 {return None;}
        let mut known=Vec::with_capacity(count);
        let(mut i,mut j)=(0,0);
        while i<left.known.len() || j<right.known.len() {
            if j==right.known.len() || (i<left.known.len() && left.known[i].0<right.known[j].0) {
                known.push(left.known[i]);i+=1;
            } else if i==left.known.len() || right.known[j].0<left.known[i].0 {
                known.push(right.known[j]);j+=1;
            } else {
                known.push((left.known[i].0,masks.union(left.known[i].1,right.known[j].1)));i+=1;j+=1;
            }
        }
        self.charge(count+1)?;
        let result=self.intern(unknown,known,masks)?;
        if self.unions.len()<262_144 {self.unions.insert(key,result);}
        Some(result)
    }
}

/// Mutates only a freshly constructed private candidate. On `None` the caller
/// discards it and rebuilds through the unchanged exact compiler; no partially
/// analyzed graph may be published. All label keys and node identities remain.
pub(super) fn restrict(states:&mut [FastBoundaryNwaState],starts:&[u32],alphabet:u32,
    masks:&mut FastBoundaryWeightInterner)->Option<Profile> {
    if states.is_empty() || states.len()>200_000 || alphabet==0 || alphabet>50_000
        || starts.iter().any(|&q|q as usize>=states.len()) {return None;}
    let order=fast_boundary_topological_order(states)?;
    let mut domains=Domains::new();let mut inputs=vec![0;states.len()];
    for &q in starts {inputs[q as usize]=1;}
    let mut p=Profile{states:states.len(),..Default::default()};
    for q in order {
        let id=inputs[q as usize];let total=domains.rows[id as usize].total;
        if total!=0 {p.live_states+=1;}
        let row=&mut states[q as usize];
        row.final_weight=masks.intersection(row.final_weight,total);
        for (target,w) in &mut row.epsilons {
            if *w==0 {continue;}
            p.edges_before+=1;
            let output=domains.meet(id,*w,masks)?;
            let new=domains.rows[output as usize].total;
            p.changed_edges+=usize::from(new!=*w);*w=new;
            if new!=0 {p.edges_after+=1;inputs[*target as usize]=domains.union(inputs[*target as usize],output,masks)?;}
        }
        for (label,branches) in &mut row.transitions {
            let push=is_negative_label(*label);
            let symbol=if push {negative_to_positive_label(*label)} else {*label};
            if symbol!=DEFAULT_LABEL && (symbol<0 || symbol as u32>=alphabet) {return None;}
            let eligible=if push || *label==DEFAULT_LABEL {total} else {domains.read(id,*label as u32,masks)};
            for (target,w) in branches {
                if *w==0 {continue;}
                p.edges_before+=1;let new=masks.intersection(*w,eligible);
                p.changed_edges+=usize::from(new!=*w);*w=new;
                if new==0 {continue;}
                p.edges_after+=1;
                let output=if push {domains.singleton(symbol as u32,new,masks)?} else {domains.unknown(new,masks)?};
                inputs[*target as usize]=domains.union(inputs[*target as usize],output,masks)?;
            }
        }
        if p.edges_before>4_000_000 || !masks.allow_work(row.epsilons.len()+row.transitions.len()+1,0,0) {return None;}
    }
    p.domains=domains.rows.len();p.stored_pairs=domains.pairs;p.work=domains.work;Some(p)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn execute(states:&[FastBoundaryNwaState],initial:&[u32],masks:&mut FastBoundaryWeightInterner)->BTreeMap<Vec<u32>,u32> {
        let mut output=BTreeMap::new();let mut todo=vec![(0u32,initial.to_vec(),1u32)];
        while let Some((q,stack,path))=todo.pop() {
            let row=&states[q as usize];let end=masks.intersection(path,row.final_weight);
            if end!=0 {let v=output.entry(stack.clone()).or_insert(0);*v=masks.union(*v,end);}
            for &(next,w) in &row.epsilons {let w=masks.intersection(path,w);if w!=0 {todo.push((next,stack.clone(),w));}}
            for (label,branches) in &row.transitions {
                let mut next_stack=stack.clone();
                if is_negative_label(*label) {next_stack.insert(0,negative_to_positive_label(*label) as u32);}
                else if next_stack.is_empty() || (*label!=DEFAULT_LABEL && next_stack[0]!=*label as u32) {continue;}
                else {next_stack.remove(0);}
                for &(next,w) in branches {let w=masks.intersection(path,w);if w!=0 {todo.push((next,next_stack.clone(),w));}}
            }
        }
        output
    }
    #[test]
    fn correlated_top_support_matches_independent_signed_execution() {
        let mut seed=9877u64;let mut next=||{seed=seed.wrapping_mul(6364136223846793005).wrapping_add(1);(seed>>32) as usize};
        let mut inputs=vec![vec![]];let mut layer=vec![vec![]];
        for _ in 0..3 {let mut new=vec![];for prefix in layer {for a in 0..3u32 {let mut v=prefix.clone();v.push(a);new.push(v);}}inputs.extend(new.clone());layer=new;}
        let mut changed=0usize;
        for case in 0..512 {
            let n=3+next()%9;let mut states=vec![];let mut masks=FastBoundaryWeightInterner::new(3,64).unwrap();
            let mut ws=vec![0,1];let mut source=FxHashMap::default();
            for pattern in [1u32,2,3,5,7] {
                let w=Weight::from_per_tsid_token_sets((0..3u32).map(|row|(row,(0..4u32).filter(|&b|(pattern.rotate_left(row)&(1<<b))!=0).collect())));
                ws.push(masks.source_weight_id(&w,&mut source,None).unwrap());
            }
            for q in 0..n {
                let mut transitions=BTreeMap::<i32,SmallVec<[(u32,u32);1]>>::new();let mut epsilons=vec![];
                for target in q+1..n {let weight=ws[next()%ws.len()];match next()%5 {
                    0=>epsilons.push((target as u32,weight)),
                    1|2=>{transitions.entry(crate::compiler::glr::labels::encode_negative_label((next()%3) as u32)).or_default().push((target as u32,weight));},
                    3=>{transitions.entry((next()%3) as i32).or_default().push((target as u32,weight));},
                    _=>{transitions.entry(DEFAULT_LABEL).or_default().push((target as u32,weight));},
                }}
                if next()%3==0 {transitions.entry(2).or_default();}
                states.push(FastBoundaryNwaState{transitions:transitions.into_iter().collect(),epsilons,final_weight:ws[next()%ws.len()]});
            }
            let mut candidate=states.clone();let profile=restrict(&mut candidate,&[0],3,&mut masks).unwrap();changed+=profile.changed_edges;
            for input in &inputs {assert_eq!(execute(&states,input,&mut masks),execute(&candidate,input,&mut masks),"case{case} stack{input:?}");}
            for (a,b) in states.iter().zip(&candidate) {assert_eq!(a.transitions.iter().map(|r|r.0).collect::<Vec<_>>(),b.transitions.iter().map(|r|r.0).collect::<Vec<_>>());}
        }
        assert!(changed>0);
    }
    #[test]
    fn finite_top_correlations_are_not_erased_at_joins_and_limits_decline() {
        let mut masks=FastBoundaryWeightInterner::new(1,64).unwrap();let mut ids=FxHashMap::default();
        let a=masks.source_weight_id(&Weight::from_token_set_for_tsid(0,[1].into_iter().collect()),&mut ids,None).unwrap();
        let b=masks.source_weight_id(&Weight::from_token_set_for_tsid(0,[2].into_iter().collect()),&mut ids,None).unwrap();
        let mut domains=Domains::new();let left=domains.singleton(0,a,&mut masks).unwrap();let right=domains.singleton(1,b,&mut masks).unwrap();
        let joined=domains.union(left,right,&mut masks).unwrap();
        assert_eq!(domains.read(joined,0,&mut masks),a);assert_eq!(domains.read(joined,1,&mut masks),b);
        let unknown=domains.unknown(a,&mut masks).unwrap();let joined=domains.union(joined,unknown,&mut masks).unwrap();
        assert_eq!(domains.read(joined,2,&mut masks),a);
        let mut bounded=Domains::new();bounded.max_rows=2;
        assert!(bounded.singleton(0,a,&mut masks).is_none());assert_eq!(bounded.rows.len(),2);
        let mut cyclic=vec![FastBoundaryNwaState{epsilons:vec![(0,1)],transitions:vec![],final_weight:1}];
        assert!(restrict(&mut cyclic,&[0],3,&mut masks).is_none());
    }
}
