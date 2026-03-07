//! Runtime serialization helpers for compiled constraints.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use range_set_blaze::RangeSetBlaze;

use crate::runtime::Constraint;

pub(crate) mod serde_vec_rsb {
    use range_set_blaze::RangeSetBlaze;
    use serde::{Deserializer, Serializer};

    pub fn serialize<S: Serializer>(
        value: &[RangeSetBlaze<u32>],
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        let _ = value;
        let _ = serializer;
        unimplemented!()
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Vec<RangeSetBlaze<u32>>, D::Error> {
        let _ = deserializer;
        unimplemented!()
    }
}

pub(crate) mod serde_vec_btmap_rsb {
    use range_set_blaze::RangeSetBlaze;
    use serde::{Deserializer, Serializer};
    use std::collections::BTreeMap;

    pub fn serialize<S: Serializer>(
        value: &[BTreeMap<u32, RangeSetBlaze<u32>>],
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        let _ = value;
        let _ = serializer;
        unimplemented!()
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Vec<BTreeMap<u32, RangeSetBlaze<u32>>>, D::Error> {
        let _ = deserializer;
        unimplemented!()
    }
}

impl Constraint {
    /// Serialize this constraint to a byte vector (bincode format).
    ///
    /// Infallible — panics only if memory is exhausted (which will crash anyway).
    pub fn save(&self) -> Vec<u8> {
        unimplemented!()
    }

    /// Deserialize a constraint from bytes (bincode format).
    pub fn load(bytes: &[u8]) -> crate::Result<Self> {
        let _ = bytes;
        unimplemented!()
    }
}
