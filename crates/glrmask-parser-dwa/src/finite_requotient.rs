//! Faithful Boolean-algebra quotient of the weights still used by a positive
//! program. Unlike support pruning, this removes no coordinate distinction
//! observable by ANY remaining input weight or its Boolean combinations.
//! Include the whole old universe (including ghost/padding coordinates).
use super::*;

pub(super) struct Decoder {
    old_rows:usize, old_tokens:usize,
    groups:Vec<Box<[(usize,u64)]>>,
    pub old_points:usize,
}
impl Decoder{
    fn decode_value(&self,value:&[u64])->Box<[u64]>{
        let mut out=vec![0u64;self.old_rows];
        for (g,members) in self.groups.iter().enumerate(){
            if value[g/64]&(1u64<<(g%64))!=0{
                for &(row,mask) in members.iter(){out[row]|=mask;}
            }
        }
        out.into_boxed_slice()
    }
    pub fn decode(&self,mut result:FiniteBoundaryDwa)->FiniteBoundaryDwa{
        result.weights=result.weights.iter().map(|v|self.decode_value(v)).collect();
        result.rows=self.old_rows;result.token_count=self.old_tokens;result
    }
    pub fn classes(&self)->usize{self.groups.len()}
}

pub(super)fn apply(states:&mut[FastBoundaryNwaState],pool:&mut FastBoundaryWeightInterner)
 ->Option<Option<Decoder>>{
    let mut used=vec![false;pool.values.len()];*used.get_mut(0)?=true;*used.get_mut(1)?=true;
    for row in states.iter(){*used.get_mut(row.final_weight as usize)?=true;
        for &(_,w) in row.epsilons.iter().chain(row.transitions.iter().flat_map(|(_,bs)|bs.iter())){
            *used.get_mut(w as usize)?=true;
        }
    }
    let active=used.iter().enumerate().filter_map(|(i,&x)|x.then_some(i)).collect::<Vec<_>>();
    let points=pool.tsid_count.checked_mul(pool.token_count)?;
    let words=active.len().div_ceil(64);
    if points.checked_mul(words)?>1_048_576{return Some(None);}
    let mut classes=FxHashMap::<Box<[u64]>,u32>::default();
    let mut representatives=Vec::new();let mut groups=Vec::<Vec<(usize,u64)>>::new();
    for point in 0..points{
        let row=point/pool.token_count;let bit=point%pool.token_count;
        let mut signature=vec![0u64;words];
        for(i,&weight)in active.iter().enumerate(){
            if pool.values[weight][row]&(1u64<<bit)!=0{signature[i/64]|=1u64<<(i%64);}
        }
        let class=if let Some(&c)=classes.get(signature.as_slice()){c}else{
            let c=groups.len()as u32;classes.insert(signature.into_boxed_slice(),c);
            groups.push(Vec::new());representatives.push((row,bit));c
        };
        let members=&mut groups[class as usize];
        if let Some((r,mask))=members.last_mut().filter(|(r,_)|*r==row){let _=r;*mask|=1u64<<bit;}
        else{members.push((row,1u64<<bit));}
    }
    let new_rows=groups.len().div_ceil(64);
    if new_rows>=pool.tsid_count{return Some(None);}
    let mut new=FastBoundaryWeightInterner::new(new_rows,64)?;
    new.limits=pool.limits;new.work=pool.work;new.borrowed_intern=pool.borrowed_intern;
    let mut map=vec![u32::MAX;pool.values.len()];
    for &weight in &active{
        let old=&pool.values[weight];let mut value:FastBoundaryWeightValue=smallvec::smallvec![0;new_rows];
        for(c,&(row,bit))in representatives.iter().enumerate(){
            if old[row]&(1u64<<bit)!=0{value[c/64]|=1u64<<(c%64);}
        }
        // New row padding duplicates class zero rather than inventing a
        // fresh ghost truth assignment. This keeps ALL/equality/subset exact.
        if groups.len()%64!=0 && value[0]&1!=0{
            value[new_rows-1]|=u64::MAX<<(groups.len()%64);
        }
        map[weight]=new.intern(value);
    }
    if new.failed{return None;}
    assert_eq!(map[0],0);assert_eq!(map[1],1);
    let decoder=Decoder{old_rows:pool.tsid_count,old_tokens:pool.token_count,
        groups:groups.into_iter().map(Vec::into_boxed_slice).collect(),old_points:points};
    for &weight in &active{if decoder.decode_value(&new.values[map[weight]as usize]).as_ref()!=pool.values[weight].as_slice(){return None;}}
    for row in states.iter_mut(){row.final_weight=map[row.final_weight as usize];
        for(_,weight)in row.epsilons.iter_mut().chain(row.transitions.iter_mut().flat_map(|(_,bs)|bs.iter_mut())){*weight=map[*weight as usize];}
    }
    *pool=new;
    Some(Some(decoder))
}

#[cfg(test)]mod tests{
 use super::*;
 #[test]fn faithful_positive_requotient_keeps_normalized_masks_and_all_bits(){
    let mut seed=251419u64;let mut next=||{seed=seed.wrapping_mul(6364136223846793005).wrapping_add(1);(seed>>32)as usize};let mut reductions=0;
    for case in 0..128{
        let rows=2+case%9;let mut pool=FastBoundaryWeightInterner::new(rows,64).unwrap();
        let mut ids=vec![0,1];
        for _ in 0..12{
            let mask=next()as u64&31;let value=(0..rows).map(|row|{
                let mut bits=0u64;for bit in 0..64{if mask&(1<<((row*64+bit)%5))!=0{bits|=1u64<<bit;}}bits
            }).collect::<FastBoundaryWeightValue>();ids.push(pool.intern(value));
        }
        let n=4+next()%9;let mut states=Vec::new();
        for q in 0..n{let mut row=FastBoundaryNwaState{epsilons:vec![],transitions:vec![],final_weight:ids[next()%ids.len()]};
            for label in [0,1,2,3,DEFAULT_LABEL]{let mut bs=SmallVec::new();for t in q+1..n{if next()%5==0{bs.push((t as u32,ids[next()%ids.len()]));}}row.transitions.push((label,bs));}
            for t in q+1..n{if next()%5==0{row.epsilons.push((t as u32,ids[next()%ids.len()]));}}
            states.push(row);
        }
        let original=states.clone();
        let before=match determinize_preconverted_small_boundary_output(&states,&[0],4,&mut pool,ids.len(),0.,Instant::now(),false,true).unwrap(){SmallBoundaryDeterminizeOutput::Finite(v)=>v.to_generic_dwa(),_=>unreachable!()};
        let decoder=apply(&mut states,&mut pool).unwrap().expect("every fixture must reduce");reductions+=1;
        // Exact graph shape, including empty label keys/targets, is unchanged.
        for(a,b)in original.iter().zip(&states){assert_eq!(a.transitions.iter().map(|(l,b)|(*l,b.iter().map(|v|v.0).collect::<Vec<_>>())).collect::<Vec<_>>(),b.transitions.iter().map(|(l,b)|(*l,b.iter().map(|v|v.0).collect::<Vec<_>>())).collect::<Vec<_>>());}
        let after=match determinize_preconverted_small_boundary_output(&states,&[0],4,&mut pool,ids.len(),0.,Instant::now(),false,true).unwrap(){SmallBoundaryDeterminizeOutput::Finite(v)=>decoder.decode(v).to_generic_dwa(),_=>unreachable!()};
        let c=crate::parser_equivalence::compare_parser_mask_prefix_languages(&before,&after,4,100_000).unwrap();assert!(c.difference.is_none(),"case{case} {:?}",c.difference);
    }
    assert_eq!(reductions,128);
 }
}
