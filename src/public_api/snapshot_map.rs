//! Small covariant persistent map for assembly snapshots.
//!
//! Unlike containers backed by interior-mutable node pools, this representation
//! preserves the covariance of borrowed grammar sources. Only an updated AVL
//! spine is copied; clone is one Arc increment. This is cold assembly metadata,
//! never part of the mask or parser runtime.
use std::{borrow::Borrow, cmp::Ordering, sync::Arc};

type Link<K,V> = Option<Arc<Node<K,V>>>;
#[derive(Debug)]
struct Node<K,V> {
    key: Arc<K>, value: V, left: Link<K,V>, right: Link<K,V>, height: u16, len: usize,
}
#[derive(Debug)]
pub(super) struct SnapshotMap<K,V> { root: Link<K,V> }
impl<K,V> Clone for SnapshotMap<K,V> {
    fn clone(&self)->Self {Self {root:self.root.clone()}}
}
impl<K,V> Default for SnapshotMap<K,V> { fn default()->Self {Self::new()} }
impl<K,V> SnapshotMap<K,V> {
    pub(super) fn new()->Self {Self {root:None}}
    pub(super) fn len(&self)->usize {Self::size(&self.root)}
    pub(super) fn is_empty(&self)->bool {self.root.is_none()}
    fn height(link:&Link<K,V>)->u16 {link.as_ref().map_or(0,|n|n.height)}
    fn size(link:&Link<K,V>)->usize {link.as_ref().map_or(0,|n|n.len)}
    fn node(key:Arc<K>,value:V,left:Link<K,V>,right:Link<K,V>)->Arc<Node<K,V>> {
        let height=1+Self::height(&left).max(Self::height(&right));
        let len=1+Self::size(&left)+Self::size(&right);
        Arc::new(Node {key,value,left,right,height,len})
    }
    pub(super) fn iter(&self)->Iter<'_,K,V> {
        let mut result=Iter {stack:Vec::with_capacity(Self::height(&self.root) as usize),remaining:self.len()};
        result.left(self.root.as_deref());result
    }
    pub(super) fn keys(&self)->impl ExactSizeIterator<Item=&K> {self.iter().map(|(k,_)|k)}
    pub(super) fn values(&self)->impl ExactSizeIterator<Item=&V> {self.iter().map(|(_,v)|v)}
}
impl<K:Ord,V> SnapshotMap<K,V> {
    pub(super) fn get<Q:?Sized+Ord>(&self,key:&Q)->Option<&V> where K:Borrow<Q> {
        let mut node=self.root.as_deref();
        while let Some(n)=node {
            match key.cmp(n.key.as_ref().borrow()) {
                Ordering::Less=>node=n.left.as_deref(),
                Ordering::Greater=>node=n.right.as_deref(),
                Ordering::Equal=>return Some(&n.value),
            }
        }
        None
    }
    pub(super) fn contains_key<Q:?Sized+Ord>(&self,key:&Q)->bool where K:Borrow<Q> {self.get(key).is_some()}
}
impl<K:Ord,V:Clone> SnapshotMap<K,V> {
    fn balanced(key:Arc<K>,value:V,left:Link<K,V>,right:Link<K,V>)->Arc<Node<K,V>> {
        let lh=Self::height(&left);let rh=Self::height(&right);
        if lh>rh+1 {
            let l=left.as_ref().expect("left-heavy AVL node");
            if Self::height(&l.left)>=Self::height(&l.right) {
                let r=Self::node(key,value,l.right.clone(),right);
                return Self::node(l.key.clone(),l.value.clone(),l.left.clone(),Some(r));
            }
            let middle=l.right.as_ref().expect("left-right AVL node");
            let a=Self::node(l.key.clone(),l.value.clone(),l.left.clone(),middle.left.clone());
            let b=Self::node(key,value,middle.right.clone(),right);
            return Self::node(middle.key.clone(),middle.value.clone(),Some(a),Some(b));
        }
        if rh>lh+1 {
            let r=right.as_ref().expect("right-heavy AVL node");
            if Self::height(&r.right)>=Self::height(&r.left) {
                let l=Self::node(key,value,left,r.left.clone());
                return Self::node(r.key.clone(),r.value.clone(),Some(l),r.right.clone());
            }
            let middle=r.left.as_ref().expect("right-left AVL node");
            let a=Self::node(key,value,left,middle.left.clone());
            let b=Self::node(r.key.clone(),r.value.clone(),middle.right.clone(),r.right.clone());
            return Self::node(middle.key.clone(),middle.value.clone(),Some(a),Some(b));
        }
        Self::node(key,value,left,right)
    }
    fn inserted(node:&Link<K,V>,key:Arc<K>,value:V)->(Arc<Node<K,V>>,Option<V>) {
        let Some(n)=node else {return (Self::node(key,value,None,None),None)};
        match key.cmp(&n.key) {
            Ordering::Less=> {
                let (left,old)=Self::inserted(&n.left,key,value);
                (Self::balanced(n.key.clone(),n.value.clone(),Some(left),n.right.clone()),old)
            }
            Ordering::Greater=> {
                let (right,old)=Self::inserted(&n.right,key,value);
                (Self::balanced(n.key.clone(),n.value.clone(),n.left.clone(),Some(right)),old)
            }
            Ordering::Equal=>(Self::node(n.key.clone(),value,n.left.clone(),n.right.clone()),Some(n.value.clone())),
        }
    }
    pub(super) fn insert(&mut self,key:K,value:V)->Option<V> {
        let (root,old)=Self::inserted(&self.root,Arc::new(key),value);self.root=Some(root);old
    }
}
pub(super) struct Iter<'a,K,V> {stack:Vec<&'a Node<K,V>>,remaining:usize}
impl<'a,K,V> Iter<'a,K,V> {
    fn left(&mut self,mut node:Option<&'a Node<K,V>>) {
        while let Some(n)=node {self.stack.push(n);node=n.left.as_deref();}
    }
}
impl<'a,K,V> Iterator for Iter<'a,K,V> {
    type Item=(&'a K,&'a V);
    fn next(&mut self)->Option<Self::Item> {
        let n=self.stack.pop()?;self.remaining-=1;self.left(n.right.as_deref());Some((n.key.as_ref(),&n.value))
    }
    fn size_hint(&self)->(usize,Option<usize>) {(self.remaining,Some(self.remaining))}
}
impl<K,V> ExactSizeIterator for Iter<'_,K,V> {}
impl<'a,K,V> IntoIterator for &'a SnapshotMap<K,V> {
    type Item=(&'a K,&'a V);type IntoIter=Iter<'a,K,V>;
    fn into_iter(self)->Self::IntoIter {self.iter()}
}
impl<K:Clone,V:Clone> IntoIterator for SnapshotMap<K,V> {
    type Item=(K,V);type IntoIter=std::vec::IntoIter<(K,V)>;
    fn into_iter(self)->Self::IntoIter {self.iter().map(|(k,v)|(k.clone(),v.clone())).collect::<Vec<_>>().into_iter()}
}
impl<K:Ord,V:Clone> FromIterator<(K,V)> for SnapshotMap<K,V> {
    fn from_iter<T:IntoIterator<Item=(K,V)>>(values:T)->Self {let mut m=Self::new();for (k,v) in values{m.insert(k,v);}m}
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    fn invariant<K:Ord+std::fmt::Debug,V>(n:&Link<K,V>)->(u16,usize) {
        let Some(n)=n else{return (0,0)};
        let (lh,ll)=invariant(&n.left);let (rh,rl)=invariant(&n.right);
        assert!(lh.abs_diff(rh)<=1,"AVL balance");assert_eq!(n.height,1+lh.max(rh));assert_eq!(n.len,1+ll+rl);
        if let Some(l)=&n.left{assert!(l.key<n.key)};if let Some(r)=&n.right{assert!(r.key>n.key)};
        (n.height,n.len)
    }
    #[test]
    fn persistent_map_order_replacement_and_snapshots_match_btree() {
        let mut a=SnapshotMap::new();let mut b=BTreeMap::new();let mut state=317u64;
        let mut snapshots=Vec::new();
        for i in 0..4096u32 {
            state=state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let key=(state>>32) as u32%513;
            if i%257==0{snapshots.push((a.clone(),b.clone()));}
            assert_eq!(a.insert(key,i),b.insert(key,i));invariant(&a.root);
            assert_eq!(a.get(&key),b.get(&key));assert_eq!(a.len(),b.len());
        }
        snapshots.push((a,b));
        for (a,b) in snapshots{assert_eq!(a.iter().collect::<Vec<_>>(),b.iter().collect::<Vec<_>>());}
    }
    #[test]
    fn persistent_map_remains_covariant_for_borrowed_sources() {
        fn narrow<'a>(m:SnapshotMap<String,&'static str>,_:&'a str)->SnapshotMap<String,&'a str>{m}
        let mut a=SnapshotMap::new();a.insert("word".to_owned(),"value");
        let local=String::new();let b=narrow(a,&local);assert_eq!(b.get("word"),Some(&"value"));
    }
}

