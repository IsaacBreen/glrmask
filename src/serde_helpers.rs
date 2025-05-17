// src/serde_helpers.rs

use bimap::BiBTreeMap;
use range_set_blaze::RangeSetBlaze;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::BTreeMap;
use std::hash::Hash;
use std::cmp::Ord;

pub mod bibtreemap_serde {
    use super::*;

    pub fn serialize<L, R, S>(map: &BiBTreeMap<L, R>, serializer: S) -> Result<S::Ok, S::Error>
    where
        L: Serialize + Ord,
        R: Serialize + Ord, // R must also be Serialize to be a value in BTreeMap<L,R>
        S: Serializer,
    {
        // Serialize the forward map (L -> R)
        let mut left_map = BTreeMap::new();
        for (l, r) in map.iter() {
            left_map.insert(l.clone(), r.clone());
        }
        left_map.serialize(serializer)
    }

    pub fn deserialize<'de, L, R, D>(deserializer: D) -> Result<BiBTreeMap<L, R>, D::Error>
    where
        L: Deserialize<'de> + Ord + Hash + Eq,
        R: Deserialize<'de> + Ord + Hash + Eq,
        D: Deserializer<'de>,
    {
        let btree_map = BTreeMap::<L, R>::deserialize(deserializer)?;
        let mut bibtree_map = BiBTreeMap::new();
        for (l, r) in btree_map {
            // insert can panic if r is already associated with a different l,
            // or l with a different r. This implies the serialized BTreeMap
            // must represent a valid bimap.
            bibtree_map.insert(l, r);
        }
        Ok(bibtree_map)
    }
}

pub mod range_set_blaze_serde {
    use std::ops::RangeInclusive;
    use super::*;

    pub fn serialize<S>(set: &RangeSetBlaze<usize>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let ranges: Vec<(usize, usize)> = set.ranges().map(|range| (*range.start(), *range.end())).collect();
        ranges.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<RangeSetBlaze<usize>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let ranges = Vec::<(usize, usize)>::deserialize(deserializer)?;
        Ok(RangeSetBlaze::from_iter(ranges.into_iter().map(|(start, end)| RangeInclusive::new(start, end))))
    }
}
