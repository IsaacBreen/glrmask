// src/serde_helpers.rs

use bimap::BiBTreeMap;
use range_set_blaze::RangeSetBlaze;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::BTreeMap;
use std::hash::Hash;
use std::cmp::Ord;
use std::clone::Clone;

pub mod bibtreemap_serde {
    use super::*; // Imports BiBTreeMap, Serialize, Serializer, Deserialize, Deserializer, Ord, Hash, Eq, Clone

    pub fn serialize<L, R, S>(map: &BiBTreeMap<L, R>, serializer: S) -> Result<S::Ok, S::Error>
    where
        L: Serialize + Clone + Ord,
        R: Serialize + Clone + Ord,
        S: Serializer,
    {
        // Serialize as a vector of (key, value) pairs
        let entries: Vec<(L, R)> = map.iter().map(|(l, r)| (l.clone(), r.clone())).collect();
        entries.serialize(serializer)
    }

    pub fn deserialize<'de, L, R, D>(deserializer: D) -> Result<BiBTreeMap<L, R>, D::Error>
    where
        L: Deserialize<'de> + Ord + Hash + Eq,
        R: Deserialize<'de> + Ord + Hash + Eq,
        D: Deserializer<'de>,
    {
        // Deserialize from a vector of (key, value) pairs
        let entries: Vec<(L, R)> = Vec::deserialize(deserializer)?;
        let mut bibtree_map = BiBTreeMap::new();
        for (l, r) in entries {
            bibtree_map.insert(l, r);
        }
        Ok(bibtree_map)
    }
}

pub mod range_set_blaze_serde {
    use std::ops::RangeInclusive;
    use super::*; // Imports RangeSetBlaze, Serialize, Serializer, Deserialize, Deserializer, Ord, Hash, Eq, Clone

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
