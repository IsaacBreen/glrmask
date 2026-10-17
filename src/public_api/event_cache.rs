//! Bounded, definition-owned exact-event component reuse.
//!
//! No cache lives on a runtime masking object or on a process-global registry.
//! A definition owns a vocabulary and this cache; the cached component does not
//! reference the definition/cache, so retaining its vocabulary creates no cycle.
use super::*;
use std::{collections::HashMap, sync::Mutex};
const CAPACITY:usize=64;
#[derive(Debug)]
pub(super) struct ExactEvent {
    ranges:Arc<[(u32,u32)]>,
    artifact:Mutex<Option<Arc<RuntimeConstraint>>>,
}
impl std::ops::Deref for ExactEvent {
    type Target=[(u32,u32)];fn deref(&self)->&Self::Target {&self.ranges}
}
impl ExactEvent {
    pub(super) fn compiled(&self,vocab:&Vocab)->Result<Arc<RuntimeConstraint>> {
        let mut artifact=self.artifact.lock().map_err(|_|Error::InternalInvariant("event adapter construction poisoned".into()))?;
        if let Some(artifact)=artifact.as_ref(){return Ok(artifact.clone());}
        let compiled=Arc::new(compile_exact_token_adapter(vocab,&self.ranges)?);
        *artifact=Some(compiled.clone());Ok(compiled)
    }
}
#[derive(Debug,Default)]
pub(super) struct ExactAdapterCache { entries:Mutex<HashMap<Arc<[(u32,u32)]>,Arc<ExactEvent>>> }
impl ExactAdapterCache {
    pub(super) fn leaf(&self,ranges:&[(u32,u32)])->Result<Arc<ExactEvent>> {
        let (leaf,evicted)={
            let mut entries=self.entries.lock().map_err(|_|Error::InternalInvariant("event adapter cache poisoned".into()))?;
            if let Some(leaf)=entries.get(ranges){return Ok(leaf.clone());}
            let evicted=if entries.len()>=CAPACITY {
                let key=entries.keys().next().cloned().expect("nonempty full cache");entries.remove(&key)
            }else{None};
            let ranges:Arc<[(u32,u32)]>=Arc::from(ranges);
            let leaf=Arc::new(ExactEvent {ranges:ranges.clone(),artifact:Mutex::new(None)});
            entries.insert(ranges,leaf.clone());(leaf,evicted)
        };
        drop(evicted);Ok(leaf)
    }
    #[cfg(test)]
    fn get(&self,vocab:&Vocab,ranges:&[(u32,u32)])->Result<Arc<RuntimeConstraint>> {self.leaf(ranges)?.compiled(vocab)}
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn same_event_set_reuses_the_compiled_component_and_preserves_identity() {
        let v=Vocab::new_with_exact_token_ids(vec![(0,b"x".to_vec())],[17,29]);
        let cache=ExactAdapterCache::default();
        let a=cache.get(&v,&[(17,17),(29,29)]).unwrap();
        let b=cache.get(&v,&[(17,17),(29,29)]).unwrap();
        assert!(Arc::ptr_eq(&a,&b));
        let mut state=a.start();
        assert!(state.mask()[0]&(1<<17)!=0);
        assert!(state.commit_bytes(b"x").is_err());
        let mut state=a.start();state.commit_token(17).unwrap();assert!(state.is_accepting());
    }
}
