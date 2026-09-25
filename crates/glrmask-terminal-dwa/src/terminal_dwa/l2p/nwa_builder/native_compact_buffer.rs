//! Stable edge-key index and contiguous mutable coefficients. This separates
//! the 12-byte lookup key from the 72-byte payload without changing any union
//! or flush order. A budget failure invalidates the entire native attempt.
use super::*;
#[derive(Default)]
pub(super) struct CompactBuffer {
    index:FxHashMap<(u32,i32,u32),u32>,
    weights:Vec<Weight>,
    pub failed:bool,
}
impl CompactBuffer {
    pub fn add(&mut self,key:(u32,i32,u32),weight:&Weight){
        if self.failed{return}
        use std::collections::hash_map::Entry;
        match self.index.entry(key){
            Entry::Occupied(entry)=>{
                let w=&mut self.weights[*entry.get()as usize];*w=w.union(weight);
            }
            Entry::Vacant(entry)=>{
                if self.weights.len()>=100_000{self.failed=true;return}
                entry.insert(self.weights.len()as u32);self.weights.push(*weight);
            }
        }
    }
    pub fn len(&self)->usize{self.index.len()}
    pub fn flush(&mut self,nwa:&mut NWA){
        if self.failed{return}
        let mut keys=std::mem::take(&mut self.index).into_iter().collect::<Vec<_>>();
        keys.sort_unstable_by_key(|(key,_)|*key);
        for((from,label,to),id)in keys{
            let w=&self.weights[id as usize];nwa.append_transition(from,label,to,w.end,w.valid,&w.bits);
        }
        self.weights.clear();
    }
}
#[cfg(test)]mod tests {
    use super::*;
    #[test]
    fn compact_event_buffer_preserves_all_key_unions_and_order(){
        let mut seed=221719u64;let mut next=||{seed=seed.wrapping_mul(6364136223846793005).wrapping_add(1);(seed>>32)as u32};
        for case in 0..128{
            let mut control=FxHashMap::<(u32,i32,u32),Weight>::default();let mut candidate=CompactBuffer::default();
            for _ in 0..1024{
                let key=(next()%24,(next()%8)as i32,next()%24);let mut bits=[0;8];bits[(next()%8)as usize]=u64::from(next());
                let w=Weight{bits,end:if case%3==0{next()%8}else{7},valid:case%7!=0};
                control.entry(key).and_modify(|old|*old=old.union(&w)).or_insert(w);candidate.add(key,&w);
            }
            let mut keys=control.keys().copied().collect::<Vec<_>>();keys.sort();let mut actual=candidate.index.keys().copied().collect::<Vec<_>>();actual.sort();assert_eq!(keys,actual);
            for key in keys{let a=control[&key];let b=candidate.weights[candidate.index[&key]as usize];assert_eq!((a.bits,a.end,a.valid),(b.bits,b.end,b.valid));}
            assert!(!candidate.failed);
        }
    }
    #[test]
    fn compact_buffer_exhaustion_is_explicit_and_no_partial_flush(){
        let mut buffer=CompactBuffer::default();buffer.weights.resize(100000,Weight::empty());
        let w=Weight{bits:[1;8],end:2,valid:true};buffer.add((0,1,2),&w);assert!(buffer.failed);
        assert!(buffer.index.is_empty());buffer.add((1,1,2),&w);assert!(buffer.index.is_empty());
    }
}
