//! Sound top-of-stack abstraction for an acyclic signed template program.
//!
//! Initial stacks are arbitrary. Epsilon preserves the abstract top set;
//! PUSH(a) produces {a}; READ(a) is impossible if a is absent, otherwise the
//! exposed tail becomes ALL. DEFAULT also forgets to ALL. In particular we
//! never apply an LR-predecessor assumption to an intermediate written stack.
//! Joining top sets loses correlations, which can only retain extra paths.
//! This is an impossibility filter, not an admission oracle or a quotient.
use super::*;

use super::signed_top_support::{StackEffect, TopDomain, TopSupportProfile};

struct Tops {
    words:usize,
    rows:Vec<Box<[u64]>>,
    ids:FxHashMap<Box<[u64]>,u32>,
    singleton:Vec<u32>,
    joins:FxHashMap<(u32,u32),u32>,
}

impl Tops {
    fn new(alphabet:usize)->Option<Self>{
        if alphabet==0||alphabet>50_000{return None;}
        let words=alphabet.div_ceil(64);
        let mut rows=Self{words,rows:Vec::new(),ids:FxHashMap::default(),
            singleton:vec![u32::MAX;alphabet],joins:FxHashMap::default()};
        rows.intern(vec![0;words])?;
        rows.intern(vec![u64::MAX;words])?;
        Some(rows)
    }
    fn intern(&mut self,bits:Vec<u64>)->Option<u32>{
        if let Some(&id)=self.ids.get(bits.as_slice()){return Some(id);}
        if self.rows.len()>=100_000||(self.rows.len()+1).checked_mul(self.words)?>4_000_000{return None;}
        let bits=bits.into_boxed_slice();let id=self.rows.len() as u32;
        self.rows.push(bits.clone());self.ids.insert(bits,id);Some(id)
    }
    fn singleton(&mut self,label:u32)->Option<u32>{
        let prior=*self.singleton.get(label as usize)?;
        if prior!=u32::MAX{return Some(prior);}
        let mut bits=vec![0;self.words];bits[label as usize/64]|=1u64<<(label%64);
        let id=self.intern(bits)?;self.singleton[label as usize]=id;Some(id)
    }
    fn contains(&self,id:u32,label:u32)->bool{
        id==1 || ((label as usize)<self.singleton.len()
            && self.rows[id as usize][label as usize/64]&(1u64<<(label%64))!=0)
    }
    fn union(&mut self,a:u32,b:u32)->Option<u32>{
        if a==b||b==0||a==1{return Some(a);}
        if a==0||b==1{return Some(b);}
        let key=if a<b{(a,b)}else{(b,a)};
        if let Some(&id)=self.joins.get(&key){return Some(id);}
        let bits=self.rows[a as usize].iter().zip(self.rows[b as usize].iter()).map(|(a,b)|a|b).collect();
        let id=self.intern(bits)?;
        if self.joins.len()<262_144{self.joins.insert(key,id);}
        Some(id)
    }
}

impl TopDomain for Tops {
    fn live(&self, input: u32) -> bool { input != 0 }
    fn final_weight(&mut self, input: u32, weight: u32) -> u32 {
        if input == 0 { 0 } else { weight }
    }
    fn epsilon(&mut self, input: u32, weight: u32) -> Option<(u32, u32)> {
        Some((input, self.final_weight(input, weight)))
    }
    fn prepare(&mut self, input: u32, effect: StackEffect) -> Option<u32> {
        if input == 0 {
            return Some(0);
        }
        match effect {
            StackEffect::Push(symbol) => self.singleton(symbol),
            StackEffect::Read(symbol) => Some(u32::from(self.contains(input, symbol))),
            StackEffect::Forget => Some(1),
        }
    }
    fn branch(&mut self, prepared: u32, _: StackEffect, weight: u32) -> Option<(u32, u32)> {
        self.epsilon(prepared, weight)
    }
    fn join(&mut self, old: u32, incoming: u32) -> Option<u32> {
        self.union(old, incoming)
    }
    fn finish_profile(&self, profile: &mut TopSupportProfile) {
        profile.top_classes = self.rows.len();
    }
}

pub(super) fn restrict(
    states: Vec<FastBoundaryNwaState>, starts: &[u32], alphabet: u32,
) -> Option<(Vec<FastBoundaryNwaState>, TopSupportProfile)> {
    super::signed_top_support::restrict(states, starts, alphabet, &mut Tops::new(alphabet as usize)?)
}

#[cfg(test)]
mod tests{
    use super::*;
    fn execute(states:&[FastBoundaryNwaState],initial:&[u32],interner:&mut FastBoundaryWeightInterner)->BTreeMap<Vec<u32>,u32>{
        let mut output=BTreeMap::new();let mut todo=vec![(0u32,initial.to_vec(),1u32)];
        while let Some((q,stack,path))=todo.pop(){
            let row=&states[q as usize];let final_weight=interner.intersection(path,row.final_weight);
            if final_weight!=0{let prior=output.entry(stack.clone()).or_insert(0);*prior=interner.union(*prior,final_weight);}
            for &(next,w) in &row.epsilons{let w=interner.intersection(path,w);if w!=0{todo.push((next,stack.clone(),w));}}
            for (label,branches) in &row.transitions{
                let mut target=stack.clone();
                if is_negative_label(*label){target.insert(0,negative_to_positive_label(*label) as u32);}
                else if target.is_empty()||(*label!=DEFAULT_LABEL&&target[0]!=*label as u32){continue;}
                else{target.remove(0);}
                for &(next,w) in branches{let w=interner.intersection(path,w);if w!=0{todo.push((next,target.clone(),w));}}
            }
        }output
    }
    #[test]
    fn signed_top_support_matches_all_small_stack_effects(){
        let mut seed=681u64;let mut next=||{seed=seed.wrapping_mul(6364136223846793005).wrapping_add(1);(seed>>32) as usize};
        let mut inputs=vec![Vec::new()];let mut layer=vec![Vec::new()];
        for _ in 0..3{let mut more=Vec::new();for prefix in layer{for a in 0..3u32{let mut v=prefix.clone();v.push(a);more.push(v);}}inputs.extend(more.clone());layer=more;}
        let mut blocked=0usize;
        for case in 0..512{
            let n=3+next()%9;let mut states=Vec::new();let mut interner=FastBoundaryWeightInterner::new(1,64).unwrap();
            let mut weights=vec![0,1];let mut source=FxHashMap::default();
            for pattern in [1u32,2,3,5]{weights.push(interner.source_weight_id(&Weight::from_token_set_for_tsid(0,(0..3u32).filter(|&b|pattern&(1<<b)!=0).collect()),&mut source,None).unwrap());}
            for q in 0..n{
                let mut transitions=BTreeMap::<i32,SmallVec<[(u32,u32);1]>>::new();let mut epsilons=Vec::new();
                for target in q+1..n{let weight=weights[next()%weights.len()];match next()%5{
                    0=>epsilons.push((target as u32,weight)),
                    1|2=>{transitions.entry(crate::compiler::glr::labels::encode_negative_label((next()%3) as u32)).or_default().push((target as u32,weight));},
                    3=>{transitions.entry((next()%3) as i32).or_default().push((target as u32,weight));},
                    _=>{transitions.entry(DEFAULT_LABEL).or_default().push((target as u32,weight));},
                }}
                states.push(FastBoundaryNwaState{transitions:transitions.into_iter().collect(),epsilons,final_weight:weights[next()%weights.len()]});
            }
            let (reduced,profile)=restrict(states.clone(),&[0],3).unwrap();blocked+=profile.blocked_labels;
            for stack in &inputs{assert_eq!(execute(&states,stack,&mut interner),execute(&reduced,stack,&mut interner),"case={case} stack={stack:?}");}
            for (original,after) in states.iter().zip(&reduced){assert_eq!(original.transitions.iter().map(|(l,_)|*l).collect::<Vec<_>>(),after.transitions.iter().map(|(l,_)|*l).collect::<Vec<_>>());}
        }
        assert!(blocked>0,"generated corpus must really prune");
    }
}
