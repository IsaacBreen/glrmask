//! Allocation preflight for untrusted assembly manifests. Check count hints
//! before reserving/decoding elements, including tiny inputs claiming huge DAGs.
use serde::{Deserialize,Deserializer,de::{self,SeqAccess,Visitor}};
use std::marker::PhantomData;
fn vector<'de,D,T,const MAX:usize>(d:D)->std::result::Result<Vec<T>,D::Error>
where D:Deserializer<'de>,T:Deserialize<'de> {
    struct Bounded<T,const MAX:usize>(PhantomData<T>);
    impl<'de,T:Deserialize<'de>,const MAX:usize> Visitor<'de> for Bounded<T,MAX> {
        type Value=Vec<T>;
        fn expecting(&self,f:&mut std::fmt::Formatter)->std::fmt::Result {write!(f,"at most {MAX} manifest elements")}
        fn visit_seq<A:SeqAccess<'de>>(self,mut seq:A)->std::result::Result<Vec<T>,A::Error> {
            if seq.size_hint().is_some_and(|n|n>MAX) {return Err(de::Error::custom("excessive artifact element count"));}
            let mut out=Vec::new();
            while let Some(value)=seq.next_element()? {
                if out.len()==MAX {return Err(de::Error::custom("excessive artifact element count"));}
                out.push(value);
            }
            Ok(out)
        }
    }
    d.deserialize_seq(Bounded::<T,MAX>(PhantomData))
}
pub(super) fn nodes<'de,D,T>(d:D)->std::result::Result<Vec<T>,D::Error>
where D:Deserializer<'de>,T:Deserialize<'de> {vector::<D,T,131_072>(d)}
pub(super) fn definitions<'de,D,T>(d:D)->std::result::Result<Vec<T>,D::Error>
where D:Deserializer<'de>,T:Deserialize<'de> {vector::<D,T,262_144>(d)}
pub(super) fn items<'de,D,T>(d:D)->std::result::Result<Vec<T>,D::Error>
where D:Deserializer<'de>,T:Deserialize<'de> {vector::<D,T,1_048_576>(d)}
