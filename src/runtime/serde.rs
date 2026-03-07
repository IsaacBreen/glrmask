
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use range_set_blaze::RangeSetBlaze;
use std::collections::BTreeMap;

use crate::runtime::Constraint;

// SEP1_MAP: this file is closest to sep1 cache serialization in
// `grammars2024/src/constraint.rs::{save_to_cache,load_from_cache}`.
// glrmask also keeps a custom nested-map serde helper here with no exact sep1
// equivalent.
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
    // SEP1_MAP: `save()` is the direct glrmask analogue of sep1
    // `GrammarConstraint::save_to_cache()` in `grammars2024/src/constraint.rs`,
    // but glrmask keeps the API byte-oriented instead of file-path-oriented.
    
    
    
    pub fn save(&self) -> Vec<u8> {
        unimplemented!()
    }

    // SEP1_MAP: `load()` is the direct glrmask analogue of sep1
    // `GrammarConstraint::load_from_cache()` in `grammars2024/src/constraint.rs`,
    // again using bytes instead of filesystem paths.
    
    pub fn load(bytes: &[u8]) -> crate::Result<Self> {
        let _ = bytes;
        unimplemented!()
    }
}
