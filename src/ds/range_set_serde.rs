//! Serde helpers for [`TokenSet`](crate::automata::weighted::weight::TokenSet).
//!
//! `RangeSetBlaze` from `range-set-blaze` does not implement `Serialize`/`Deserialize`.
//! This module provides helper sub-modules for use with `#[serde(with = "...")]` on
//! fields that contain `TokenSet` in various container shapes.
//!
//! Wire format: a `TokenSet` is serialised as a flat `Vec<u32>` of
//! `[lo₀, hi₀, lo₁, hi₁, …]` inclusive-range endpoints, matching the format
//! previously used by the deleted `RangeSet` wrapper.

use crate::automata::weighted::weight::TokenSet;

// ------------------------------------------------------------------
// Bare TokenSet
// ------------------------------------------------------------------

/// `#[serde(with = "crate::ds::range_set_serde::bare")]`
#[allow(dead_code)]
pub mod bare {
    use super::*;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(rs: &TokenSet, s: S) -> Result<S::Ok, S::Error> {
        let flat: Vec<u32> = rs.ranges().flat_map(|r| [*r.start(), *r.end()]).collect();
        flat.serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<TokenSet, D::Error> {
        let flat = Vec::<u32>::deserialize(d)?;
        Ok(flat.chunks_exact(2).map(|c| c[0]..=c[1]).collect())
    }
}

// ------------------------------------------------------------------
// Vec<TokenSet>
// ------------------------------------------------------------------

/// `#[serde(with = "crate::ds::range_set_serde::vec_rsb")]`
pub mod vec_rsb {
    use super::*;
    use serde::ser::SerializeSeq;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(v: &[TokenSet], s: S) -> Result<S::Ok, S::Error> {
        let mut seq = s.serialize_seq(Some(v.len()))?;
        for rs in v {
            let flat: Vec<u32> = rs.ranges().flat_map(|r| [*r.start(), *r.end()]).collect();
            seq.serialize_element(&flat)?;
        }
        seq.end()
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<TokenSet>, D::Error> {
        let outer = Vec::<Vec<u32>>::deserialize(d)?;
        Ok(outer
            .into_iter()
            .map(|flat| flat.chunks_exact(2).map(|c| c[0]..=c[1]).collect())
            .collect())
    }

    pub(super) fn rsb_from_flat(flat: Vec<u32>) -> TokenSet {
        flat.chunks_exact(2).map(|c| c[0]..=c[1]).collect()
    }
}

// ------------------------------------------------------------------
// Vec<BTreeMap<u32, TokenSet>>
// ------------------------------------------------------------------

/// `#[serde(with = "crate::ds::range_set_serde::vec_btmap_rsb")]`
pub mod vec_btmap_rsb {
    use super::vec_rsb::rsb_from_flat;
    use crate::automata::weighted::weight::TokenSet;
    use serde::ser::SerializeSeq;
    use serde::{Deserialize, Deserializer, Serializer};
    use std::collections::BTreeMap;

    pub fn serialize<S: Serializer>(
        v: &[BTreeMap<u32, TokenSet>],
        s: S,
    ) -> Result<S::Ok, S::Error> {
        let mut seq = s.serialize_seq(Some(v.len()))?;
        for map in v {
            let proxy: BTreeMap<u32, Vec<u32>> = map
                .iter()
                .map(|(&k, rs)| {
                    let flat: Vec<u32> =
                        rs.ranges().flat_map(|r| [*r.start(), *r.end()]).collect();
                    (k, flat)
                })
                .collect();
            seq.serialize_element(&proxy)?;
        }
        seq.end()
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<Vec<BTreeMap<u32, TokenSet>>, D::Error> {
        let outer = Vec::<BTreeMap<u32, Vec<u32>>>::deserialize(d)?;
        Ok(outer
            .into_iter()
            .map(|proxy| {
                proxy
                    .into_iter()
                    .map(|(k, flat)| (k, rsb_from_flat(flat)))
                    .collect()
            })
            .collect())
    }
}
