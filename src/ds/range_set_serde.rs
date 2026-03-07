//! Serde helpers for [`TokenSet`](crate::automata::weighted::weight::TokenSet).
//!
//! `RangeSetBlaze` from `range-set-blaze` does not implement `Serialize`/`Deserialize`.
//! This module provides helper sub-modules for use with `#[serde(with = "...")]` on
//! fields that contain `TokenSet` in various container shapes.
//!
//! Wire format: a `TokenSet` is serialised as a flat `Vec<u32>` of
//! `[lo₀, hi₀, lo₁, hi₁, …]` inclusive-range endpoints, matching the format
//! previously used by the deleted `RangeSet` wrapper.
#![allow(unused_imports, unused_variables, dead_code)]
#![allow(unused_imports, unused_variables, unused_mut, dead_code)]

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
        unimplemented!("cargo-check-only stub")
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<TokenSet, D::Error> {
        unimplemented!("cargo-check-only stub")
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
        unimplemented!("cargo-check-only stub")
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<TokenSet>, D::Error> {
        unimplemented!("cargo-check-only stub")
    }

    pub(super) fn rsb_from_flat(flat: Vec<u32>) -> TokenSet {
        unimplemented!("cargo-check-only stub")
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
        unimplemented!("cargo-check-only stub")
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<Vec<BTreeMap<u32, TokenSet>>, D::Error> {
        unimplemented!("cargo-check-only stub")
    }
}
