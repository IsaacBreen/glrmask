
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use range_set_blaze::RangeSetBlaze;
use std::collections::BTreeMap;

use crate::runtime::Constraint;

pub(in crate::runtime) mod serde_nested_btmap_rsb {
    use range_set_blaze::RangeSetBlaze;
    use serde::{Deserializer, Serializer};
    use std::collections::BTreeMap;

    pub(in crate::runtime) fn serialize<S: Serializer>(
        value: &BTreeMap<u32, BTreeMap<u32, BTreeMap<u32, RangeSetBlaze<u32>>>>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        let _ = value;
        let _ = serializer;
        unimplemented!()
    }

    pub(in crate::runtime) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<BTreeMap<u32, BTreeMap<u32, BTreeMap<u32, RangeSetBlaze<u32>>>>, D::Error> {
        let _ = deserializer;
        unimplemented!()
    }
}

impl Constraint {
    
    
    
    pub fn save(&self) -> Vec<u8> {
        unimplemented!()
    }

    
    pub fn load(bytes: &[u8]) -> crate::Result<Self> {
        let _ = bytes;
        unimplemented!()
    }
}
