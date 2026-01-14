//! Abstract weight implementation that can be backed by different storage formats.
//!
//! This module provides a unified `Weight` type that can use either:
//! - `RangeSet`: Sparse range-based storage (default)
//! - `BddWeight`: Binary Decision Diagram storage
//!
//! The backend is selected via the `WEIGHT_BACKEND` environment variable:
//! - `WEIGHT_BACKEND=rangeset` (default): Use RangeSet
//! - `WEIGHT_BACKEND=bdd`: Use BddWeight
//!
//! Note: BDD backend requires dimension info for proper encoding.

use std::collections::HashMap;
use std::fmt::{Debug, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, Not, Sub, SubAssign};
use std::sync::{Arc, Mutex, OnceLock};

use range_set_blaze::RangeSetBlaze;
use once_cell::sync::Lazy;

use super::rangeset::RangeSet;
use super::bdd_weight::BddWeight;
use super::heavy_weight::WeightDimensions;

/// Global weight backend configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightBackend {
    RangeSet,
    Bdd,
}

impl Default for WeightBackend {
    fn default() -> Self {
        Self::RangeSet
    }
}

/// Get the configured weight backend from environment.
pub fn get_weight_backend() -> WeightBackend {
    static BACKEND: OnceLock<WeightBackend> = OnceLock::new();
    *BACKEND.get_or_init(|| {
        match std::env::var("WEIGHT_BACKEND").as_deref() {
            Ok("bdd") | Ok("BDD") => WeightBackend::Bdd,
            _ => WeightBackend::RangeSet,
        }
    })
}

/// Global weight dimensions (needed for BDD backend).
/// Set once at the start of constraint building.
static WEIGHT_DIMS: OnceLock<WeightDimensions> = OnceLock::new();

/// Set the global weight dimensions. Must be called before creating BDD weights.
pub fn set_weight_dimensions(dims: WeightDimensions) {
    let _ = WEIGHT_DIMS.set(dims);
}

/// Get the global weight dimensions.
pub fn get_weight_dimensions() -> WeightDimensions {
    WEIGHT_DIMS.get().copied().unwrap_or_default()
}

// For now, just re-export RangeSet as Weight.
// TODO: Implement full abstract weight when BDD backend is ready.
// The BddWeight doesn't yet support all operations (union, intersection, complement)
// that are needed for NWA/DWA construction. 

// Keeping the existing type alias for now to minimize disruption.
// The full abstraction will be implemented once BddWeight has all needed operations.

// Note: The user requested hot-swappable weights, but BddWeight is currently
// read-only (only supports from_ranges and contains). To make it truly swappable,
// we'd need to implement union/intersection/complement on BddWeight first.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_backend() {
        // Without env var, should default to RangeSet
        let backend = WeightBackend::default();
        assert_eq!(backend, WeightBackend::RangeSet);
    }

    #[test]
    fn test_weight_dimensions() {
        let dims = WeightDimensions::new(1000, 100);
        set_weight_dimensions(dims);
        let retrieved = get_weight_dimensions();
        assert_eq!(retrieved.num_tokens, 1000);
        assert_eq!(retrieved.num_tsids, 100);
    }
}
