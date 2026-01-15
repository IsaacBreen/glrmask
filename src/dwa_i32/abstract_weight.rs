//! Abstract weight type that can be backed by different storage formats.
//!
//! This module provides an `AbstractWeight` enum that wraps either:
//! - `RangeSet`: Sparse range-based storage (default, fast operations, good for sparse weights)
//! - `BddWeight`: Binary Decision Diagram storage (custom 5-byte nodes, memory efficient)
//! - `BddWeightBiodivine`: BDD storage using biodivine_lib_bdd (battle-tested, fewer nodes)
//! - `FactoredWeight`: 2D factored representation (union of Cartesian products)
//! - `FactoredValidate`: factored + RangeSet with validation
//!
//! The backend is selected at compile time via the `WEIGHT_BACKEND` environment variable,
//! checked once at startup. See `get_weight_backend()` for details.
//!
//! # Usage
//!
//! The `AbstractWeight` type is a drop-in replacement for `RangeSet`:
//! ```ignore
//! use sep1::dwa_i32::AbstractWeight;
//!
//! let w1 = AbstractWeight::from_item(5);
//! let w2 = AbstractWeight::from_ranges(&[(10, 20)]);
//! let union = &w1 | &w2;
//! let intersect = &w1 & &w2;
//! ```
//!
//! # Backend Selection
//!
//! - `WEIGHT_BACKEND=rangeset` (default): Uses RangeSet with interning
//! - `WEIGHT_BACKEND=bdd`: Uses custom BddWeight with 5-byte nodes (memory efficient)
//! - `WEIGHT_BACKEND=bdd-biodivine`: Uses biodivine_lib_bdd (battle-tested, more features)
//! - `WEIGHT_BACKEND=factored`: Uses 2D factored representation (experimental)
//! - `WEIGHT_BACKEND=factored-validate`: Uses factored + RangeSet with validation

use std::fmt::{Debug, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, Not, Sub, SubAssign};
use std::sync::{Arc, OnceLock};
use std::iter::FromIterator;

use range_set_blaze::RangeSetBlaze;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::rangeset::RangeSet;
use super::bdd_weight::BddWeight;
use super::bdd_weight_biodivine::BddWeightBiodivine;
use super::factored_weight::FactoredWeight;
use super::factored_validate_weight::FactoredValidateWeight;
use super::heavy_weight::WeightDimensions;

// ============================================================================
// Backend Configuration
// ============================================================================

/// Weight storage backend options.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightBackend {
    /// Sparse range-based storage (default).
    RangeSet,
    /// Binary Decision Diagram storage (custom 5-byte nodes, memory efficient).
    Bdd,
    /// Binary Decision Diagram using biodivine_lib_bdd (battle-tested, more features).
    BddBiodivine,
    /// 2D factored representation (union of Cartesian products).
    Factored,
    /// Factored + RangeSet with validation.
    FactoredValidate,
}

impl Default for WeightBackend {
    fn default() -> Self {
        Self::RangeSet
    }
}

/// Get the configured weight backend from environment.
/// 
/// Reads `WEIGHT_BACKEND` env var once and caches the result.
/// Returns `RangeSet` by default.
/// 
/// Options:
/// - `rangeset` (default): RangeSet with interning
/// - `bdd`: Custom BDD with 5-byte nodes (memory efficient)
/// - `bdd-biodivine`: biodivine_lib_bdd (battle-tested, fewer nodes but more memory)
/// - `factored`: 2D factored representation (experimental)
pub fn get_weight_backend() -> WeightBackend {
    static BACKEND: OnceLock<WeightBackend> = OnceLock::new();
    *BACKEND.get_or_init(|| {
        match std::env::var("WEIGHT_BACKEND").as_deref() {
            Ok("bdd") | Ok("BDD") => WeightBackend::Bdd,
            Ok("bdd-biodivine") | Ok("BDD-BIODIVINE") | Ok("biodivine") => WeightBackend::BddBiodivine,
            Ok("factored") | Ok("FACTORED") => WeightBackend::Factored,
            Ok("factored-validate") | Ok("FACTORED-VALIDATE") | Ok("factored_validate") | Ok("FACTORED_VALIDATE") => {
                WeightBackend::FactoredValidate
            }
            _ => WeightBackend::RangeSet,
        }
    })
}

/// Global weight dimensions for BDD backend.
static WEIGHT_DIMS: OnceLock<WeightDimensions> = OnceLock::new();

/// Set the global weight dimensions. Must be called before creating BDD weights.
pub fn set_weight_dimensions(dims: WeightDimensions) {
    let _ = WEIGHT_DIMS.set(dims);
}

/// Get the global weight dimensions.
pub fn get_weight_dimensions() -> WeightDimensions {
    WEIGHT_DIMS.get().copied().unwrap_or_default()
}

// ============================================================================
// AbstractWeight
// ============================================================================

/// Abstract weight type that can use different storage backends.
/// 
/// This is a thin wrapper around either `RangeSet` (default), `BddWeight`, `BddWeightBiodivine`, or `FactoredWeight`.
/// All operations delegate to the underlying implementation.
#[derive(Clone)]
pub enum AbstractWeight {
    /// RangeSet backend (default, interned).
    Rs(RangeSet),
    /// Custom BddWeight backend (5-byte nodes, memory efficient).
    Bdd(Arc<BddWeight>),
    /// Biodivine BddWeight backend (battle-tested, fewer nodes).
    BddBiodivine(Arc<BddWeightBiodivine>),
    /// 2D factored representation (union of Cartesian products).
    Factored(Arc<FactoredWeight>),
    /// Factored + RangeSet with validation.
    FactoredValidate(Arc<FactoredValidateWeight>),
}

impl AbstractWeight {
    // ------------------------------------------------------------------------
    // Constructors
    // ------------------------------------------------------------------------

    /// Create an empty weight (accepts nothing).
    pub fn zeros() -> Self {
        match get_weight_backend() {
            WeightBackend::RangeSet => Self::Rs(RangeSet::zeros()),
            WeightBackend::Bdd => {
                let dims = get_weight_dimensions();
                Self::Bdd(Arc::new(BddWeight::empty(dims.num_tsids as u16, dims.num_tokens as u16)))
            }
            WeightBackend::BddBiodivine => {
                let dims = get_weight_dimensions();
                Self::BddBiodivine(Arc::new(BddWeightBiodivine::empty(dims.num_tsids as u16, dims.num_tokens as u16)))
            }
            WeightBackend::Factored => {
                let dims = get_weight_dimensions();
                Self::Factored(Arc::new(FactoredWeight::empty(dims.num_tsids as u16)))
            }
            WeightBackend::FactoredValidate => {
                let dims = get_weight_dimensions();
                let factored = FactoredWeight::empty(dims.num_tsids as u16);
                let rangeset = RangeSet::zeros();
                Self::FactoredValidate(Arc::new(FactoredValidateWeight::new(factored, rangeset)))
            }
        }
    }

    /// Create a full weight (accepts everything).
    pub fn all() -> Self {
        match get_weight_backend() {
            WeightBackend::RangeSet => Self::Rs(RangeSet::all()),
            WeightBackend::Bdd => {
                let dims = get_weight_dimensions();
                Self::Bdd(Arc::new(BddWeight::full(dims.num_tsids as u16, dims.num_tokens as u16)))
            }
            WeightBackend::BddBiodivine => {
                let dims = get_weight_dimensions();
                Self::BddBiodivine(Arc::new(BddWeightBiodivine::full(dims.num_tsids as u16, dims.num_tokens as u16)))
            }
            WeightBackend::Factored => {
                let dims = get_weight_dimensions();
                // Full weight: all tokens × all tsids
                let mut tokens = RangeSetBlaze::new();
                tokens.ranges_insert(0..=(dims.num_tokens as u16 - 1));
                let mut tsids = RangeSetBlaze::new();
                tsids.ranges_insert(0..=(dims.num_tsids as u16 - 1));
                Self::Factored(Arc::new(FactoredWeight::from_product(tokens, tsids, dims.num_tsids as u16)))
            }
            WeightBackend::FactoredValidate => {
                let dims = get_weight_dimensions();
                let mut tokens = RangeSetBlaze::new();
                tokens.ranges_insert(0..=(dims.num_tokens as u16 - 1));
                let mut tsids = RangeSetBlaze::new();
                tsids.ranges_insert(0..=(dims.num_tsids as u16 - 1));
                let factored = FactoredWeight::from_product(tokens, tsids, dims.num_tsids as u16);
                let rangeset = RangeSet::all();
                Self::FactoredValidate(Arc::new(FactoredValidateWeight::new(factored, rangeset)))
            }
        }
    }

    /// Create a weight containing a single position.
    pub fn from_item(item: usize) -> Self {
        match get_weight_backend() {
            WeightBackend::RangeSet => Self::Rs(RangeSet::from_item(item)),
            WeightBackend::Bdd => {
                let dims = get_weight_dimensions();
                Self::Bdd(Arc::new(BddWeight::from_ranges(
                    std::iter::once((item, item)),
                    dims.num_tsids as u16,
                    dims.num_tokens as u16,
                )))
            }
            WeightBackend::BddBiodivine => {
                let dims = get_weight_dimensions();
                Self::BddBiodivine(Arc::new(BddWeightBiodivine::from_ranges(
                    std::iter::once((item, item)),
                    dims.num_tsids as u16,
                    dims.num_tokens as u16,
                )))
            }
            WeightBackend::Factored => {
                let dims = get_weight_dimensions();
                Self::Factored(Arc::new(FactoredWeight::from_1d_ranges(
                    std::iter::once((item, item)),
                    dims.num_tsids as usize,
                )))
            }
            WeightBackend::FactoredValidate => {
                let dims = get_weight_dimensions();
                let factored = FactoredWeight::from_1d_ranges(
                    std::iter::once((item, item)),
                    dims.num_tsids as usize,
                );
                let rangeset = RangeSet::from_item(item);
                Self::FactoredValidate(Arc::new(FactoredValidateWeight::new(factored, rangeset)))
            }
        }
    }

    /// Create a weight from ranges.
    pub fn from_ranges(ranges: &[(usize, usize)]) -> Self {
        match get_weight_backend() {
            WeightBackend::RangeSet => Self::Rs(RangeSet::from_ranges(ranges)),
            WeightBackend::Bdd => {
                let dims = get_weight_dimensions();
                Self::Bdd(Arc::new(BddWeight::from_ranges(
                    ranges.iter().copied(),
                    dims.num_tsids as u16,
                    dims.num_tokens as u16,
                )))
            }
            WeightBackend::BddBiodivine => {
                let dims = get_weight_dimensions();
                Self::BddBiodivine(Arc::new(BddWeightBiodivine::from_ranges(
                    ranges.iter().copied(),
                    dims.num_tsids as u16,
                    dims.num_tokens as u16,
                )))
            }
            WeightBackend::Factored => {
                let dims = get_weight_dimensions();
                Self::Factored(Arc::new(FactoredWeight::from_1d_ranges(
                    ranges.iter().copied(),
                    dims.num_tsids as usize,
                )))
            }
            WeightBackend::FactoredValidate => {
                let dims = get_weight_dimensions();
                let factored = FactoredWeight::from_1d_ranges(ranges.iter().copied(), dims.num_tsids as usize);
                let rangeset = RangeSet::from_ranges(ranges);
                Self::FactoredValidate(Arc::new(FactoredValidateWeight::new(factored, rangeset)))
            }
        }
    }

    /// Create a weight from a RangeSetBlaze.
    pub fn from_rsb(rsb: RangeSetBlaze<usize>) -> Self {
        match get_weight_backend() {
            WeightBackend::RangeSet => Self::Rs(RangeSet::from_rsb(rsb)),
            WeightBackend::Bdd => {
                let dims = get_weight_dimensions();
                let ranges: Vec<(usize, usize)> = rsb.ranges().map(|r| (*r.start(), *r.end())).collect();
                Self::Bdd(Arc::new(BddWeight::from_ranges(
                    ranges.into_iter(),
                    dims.num_tsids as u16,
                    dims.num_tokens as u16,
                )))
            }
            WeightBackend::BddBiodivine => {
                let dims = get_weight_dimensions();
                let ranges: Vec<(usize, usize)> = rsb.ranges().map(|r| (*r.start(), *r.end())).collect();
                Self::BddBiodivine(Arc::new(BddWeightBiodivine::from_ranges(
                    ranges.into_iter(),
                    dims.num_tsids as u16,
                    dims.num_tokens as u16,
                )))
            }
            WeightBackend::Factored => {
                let dims = get_weight_dimensions();
                let ranges: Vec<(usize, usize)> = rsb.ranges().map(|r| (*r.start(), *r.end())).collect();
                Self::Factored(Arc::new(FactoredWeight::from_1d_ranges(
                    ranges.into_iter(),
                    dims.num_tsids as usize,
                )))
            }
            WeightBackend::FactoredValidate => {
                let dims = get_weight_dimensions();
                let ranges: Vec<(usize, usize)> = rsb.ranges().map(|r| (*r.start(), *r.end())).collect();
                let factored = FactoredWeight::from_1d_ranges(ranges.into_iter(), dims.num_tsids as usize);
                let rangeset = RangeSet::from_rsb(rsb);
                Self::FactoredValidate(Arc::new(FactoredValidateWeight::new(factored, rangeset)))
            }
        }
    }
    
    /// Create a weight from a token set (N-space) × all TSIDs.
    /// This is efficient for weight-heavy mode where we have tokens in N-space
    /// and want the weight to cover all TSID values.
    /// 
    /// For FactoredWeight, this is O(n_ranges) instead of O(n_tokens × n_tsids).
    pub fn from_token_set_all_tsids(tokens: RangeSetBlaze<usize>) -> Self {
        let dims = get_weight_dimensions();
        match get_weight_backend() {
            WeightBackend::Factored => {
                Self::Factored(Arc::new(FactoredWeight::from_token_set_all_tsids(
                    tokens,
                    dims.num_tsids as u16,
                )))
            }
            WeightBackend::FactoredValidate => {
                let factored = FactoredWeight::from_token_set_all_tsids(tokens.clone(), dims.num_tsids as u16);
                let expanded = crate::dwa_i32::weight_expansion::expand_rsb(&tokens, dims.num_tsids);
                let rangeset = RangeSet::from_rsb(expanded);
                Self::FactoredValidate(Arc::new(FactoredValidateWeight::new(factored, rangeset)))
            }
            // For other backends, fall back to expanding to N×M space
            _ => {
                let expanded = crate::dwa_i32::weight_expansion::expand_rsb(&tokens, dims.num_tsids);
                Self::from_rsb(expanded)
            }
        }
    }
    
    /// Create a weight from a token set (N-space) × specific TSID.
    /// This is efficient for weight-heavy precomputation where we know the exact tsid.
    /// 
    /// For FactoredWeight, this is O(n_ranges) instead of O(n_tokens × n_tsids).
    pub fn from_token_set_specific_tsid(tokens: RangeSetBlaze<usize>, tsid: usize) -> Self {
        let dims = get_weight_dimensions();
        match get_weight_backend() {
            WeightBackend::Factored => {
                Self::Factored(Arc::new(FactoredWeight::from_token_set_specific_tsid(
                    tokens,
                    tsid,
                    dims.num_tsids as u16,
                )))
            }
            WeightBackend::FactoredValidate => {
                let factored = FactoredWeight::from_token_set_specific_tsid(tokens.clone(), tsid, dims.num_tsids as u16);
                let num_tsids = dims.num_tsids;
                let positions: RangeSetBlaze<usize> = tokens.iter()
                    .map(|t| t * num_tsids + tsid)
                    .collect();
                let rangeset = RangeSet::from_rsb(positions);
                Self::FactoredValidate(Arc::new(FactoredValidateWeight::new(factored, rangeset)))
            }
            // For other backends, fall back to explicit 1D expansion
            _ => {
                // Create 1D positions: token * num_tsids + tsid for each token
                let num_tsids = dims.num_tsids;
                let positions: RangeSetBlaze<usize> = tokens.iter()
                    .map(|t| t * num_tsids + tsid)
                    .collect();
                Self::from_rsb(positions)
            }
        }
    }

    /// Create a weight for positions 0..=len-1.
    pub fn ones(len: usize) -> Self {
        match get_weight_backend() {
            WeightBackend::RangeSet => Self::Rs(RangeSet::ones(len)),
            WeightBackend::Bdd => {
                let dims = get_weight_dimensions();
                Self::Bdd(Arc::new(BddWeight::from_ranges(
                    std::iter::once((0, len.saturating_sub(1))),
                    dims.num_tsids as u16,
                    dims.num_tokens as u16,
                )))
            }
            WeightBackend::BddBiodivine => {
                let dims = get_weight_dimensions();
                Self::BddBiodivine(Arc::new(BddWeightBiodivine::from_ranges(
                    std::iter::once((0, len.saturating_sub(1))),
                    dims.num_tsids as u16,
                    dims.num_tokens as u16,
                )))
            }
            WeightBackend::Factored => {
                let dims = get_weight_dimensions();
                Self::Factored(Arc::new(FactoredWeight::from_1d_ranges(
                    std::iter::once((0, len.saturating_sub(1))),
                    dims.num_tsids as usize,
                )))
            }
            WeightBackend::FactoredValidate => {
                let dims = get_weight_dimensions();
                let factored = FactoredWeight::from_1d_ranges(
                    std::iter::once((0, len.saturating_sub(1))),
                    dims.num_tsids as usize,
                );
                let rangeset = RangeSet::ones(len);
                Self::FactoredValidate(Arc::new(FactoredValidateWeight::new(factored, rangeset)))
            }
        }
    }

    /// Create a weight representing all tokens for specific TSID columns.
    /// This is optimized for BDD backends where it's O(|tsids|) instead of O(|tsids| * |tokens|).
    ///
    /// For weight-heavy mode: represents {t*M + s : t in 0..N, s in tsids} where M = num_tsids, N = num_tokens
    ///
    /// # Arguments
    /// * `tsids` - Iterator of TSID values to include
    /// * `num_tsids` - Total number of TSIDs (M dimension)
    /// * `num_tokens` - Total number of LLM tokens (N dimension)
    pub fn tsid_columns_with_dims<I: IntoIterator<Item = usize>>(
        tsids: I,
        num_tsids: usize,
        num_tokens: usize,
    ) -> Self {
        match get_weight_backend() {
            WeightBackend::RangeSet => {
                // For RangeSet, build ranges efficiently
                let tsids: Vec<usize> = tsids.into_iter().collect();
                
                let mut rsb = RangeSetBlaze::new();
                for n in 0..num_tokens {
                    let offset = n * num_tsids;
                    for &tsid in &tsids {
                        rsb.insert(tsid + offset);
                    }
                }
                Self::Rs(RangeSet::from_rsb(rsb))
            }
            WeightBackend::Bdd => {
                // For custom BDD, build via RangeSet
                let tsids: Vec<usize> = tsids.into_iter().collect();
                
                let mut rsb = RangeSetBlaze::new();
                for n in 0..num_tokens {
                    let offset = n * num_tsids;
                    for &tsid in &tsids {
                        rsb.insert(tsid + offset);
                    }
                }
                Self::Bdd(Arc::new(BddWeight::from_ranges(
                    rsb.ranges().map(|r| (*r.start(), *r.end())),
                    num_tsids as u16,
                    num_tokens as u16,
                )))
            }
            WeightBackend::BddBiodivine => {
                // Optimized: use BDD structure directly
                Self::BddBiodivine(Arc::new(BddWeightBiodivine::tsid_columns(
                    tsids.into_iter().map(|t| t as u16),
                    num_tsids as u16,
                    num_tokens as u16,
                )))
            }
            WeightBackend::Factored => {
                // For FactoredWeight, create a product of all tokens × specified tsids
                let tsid_set: RangeSetBlaze<u16> = tsids.into_iter().map(|t| t as u16).collect();
                let token_set: RangeSetBlaze<u16> = (0..num_tokens as u16).collect();
                Self::Factored(Arc::new(FactoredWeight::from_product(token_set, tsid_set, num_tsids as u16)))
            }
            WeightBackend::FactoredValidate => {
                let tsid_vec: Vec<usize> = tsids.into_iter().collect();
                let tsid_set: RangeSetBlaze<u16> = tsid_vec.iter().map(|t| *t as u16).collect();
                let token_set: RangeSetBlaze<u16> = (0..num_tokens as u16).collect();
                let factored = FactoredWeight::from_product(token_set, tsid_set, num_tsids as u16);
                let mut rsb = RangeSetBlaze::new();
                for n in 0..num_tokens {
                    let offset = n * num_tsids;
                    for &tsid in &tsid_vec {
                        rsb.insert(tsid + offset);
                    }
                }
                let rangeset = RangeSet::from_rsb(rsb);
                Self::FactoredValidate(Arc::new(FactoredValidateWeight::new(factored, rangeset)))
            }
        }
    }
    
    /// Create a weight representing all tokens for specific TSID columns.
    /// Uses global weight dimensions from `get_weight_dimensions()`.
    /// This is optimized for BDD backends where it's O(|tsids|) instead of O(|tsids| * |tokens|).
    pub fn tsid_columns<I: IntoIterator<Item = usize>>(tsids: I) -> Self {
        let dims = get_weight_dimensions();
        Self::tsid_columns_with_dims(tsids, dims.num_tsids, dims.num_tokens)
    }

    // ------------------------------------------------------------------------
    // Queries
    // ------------------------------------------------------------------------

    /// Check if this weight is empty (accepts nothing).
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Rs(rs) => rs.is_empty(),
            Self::Bdd(bdd) => bdd.is_empty(),
            Self::BddBiodivine(bdd) => bdd.is_empty(),
            Self::Factored(fw) => fw.is_empty(),
            Self::FactoredValidate(fw) => fw.rangeset().is_empty(),
        }
    }

    /// Fast check if this weight is "all" (accepts everything).
    pub fn is_all_fast(&self) -> bool {
        match self {
            Self::Rs(rs) => rs.is_all_fast(),
            Self::Bdd(bdd) => bdd.is_full(),
            Self::BddBiodivine(bdd) => bdd.is_full(),
            Self::Factored(fw) => fw.is_full(),
            Self::FactoredValidate(fw) => fw.rangeset().is_all_fast(),
        }
    }

    /// Check if a position is contained in this weight.
    pub fn contains(&self, pos: usize) -> bool {
        match self {
            Self::Rs(rs) => rs.contains(pos),
            Self::Bdd(bdd) => bdd.contains_pos(pos),
            Self::BddBiodivine(bdd) => bdd.contains_pos(pos),
            Self::Factored(fw) => fw.contains_pos(pos),
            Self::FactoredValidate(fw) => fw.rangeset().contains(pos),
        }
    }

    /// Check if two weights are disjoint.
    pub fn is_disjoint(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Rs(a), Self::Rs(b)) => a.is_disjoint(b),
            (Self::Bdd(a), Self::Bdd(b)) => a.intersection(b).is_empty(),
            (Self::Factored(a), Self::Factored(b)) => a.intersection(b).is_empty(),
            (Self::FactoredValidate(a), Self::FactoredValidate(b)) => a.rangeset().is_disjoint(b.rangeset()),
            _ => self.to_rangeset().is_disjoint(&other.to_rangeset()),
        }
    }

    /// Get the number of ranges (only meaningful for RangeSet backend).
    /// For Factored backend, returns total_ranges() which is an approximation.
    pub fn num_ranges(&self) -> usize {
        match self {
            Self::Rs(rs) => rs.num_ranges(),
            Self::Bdd(bdd) => bdd.to_rangeset().ranges_len(),
            Self::BddBiodivine(bdd) => bdd.to_rangeset().ranges_len(),
            Self::Factored(fw) => fw.total_ranges(),
            Self::FactoredValidate(fw) => fw.rangeset().num_ranges(),
        }
    }

    /// Get the number of elements in this weight.
    pub fn len(&self) -> usize {
        match self {
            Self::Rs(rs) => rs.len(),
            Self::Bdd(bdd) => bdd.len(),
            Self::BddBiodivine(bdd) => bdd.len(),
            Self::Factored(fw) => fw.len(),
            Self::FactoredValidate(fw) => fw.rangeset().len(),
        }
    }

    /// Returns the complement of this weight.
    pub fn complement(&self) -> Self {
        !self
    }

    /// Get a fast hash/fingerprint for this weight.
    /// Used for quick equality checking in interning.
    pub fn fp(&self) -> u64 {
        match self {
            Self::Rs(rs) => rs.fast_hash(),
            Self::Bdd(_) | Self::BddBiodivine(_) | Self::Factored(_) | Self::FactoredValidate(_) => {
                // For BDD/Factored, compute a hash of the ranges
                use std::hash::{Hash, Hasher};
                use std::collections::hash_map::DefaultHasher;
                let mut hasher = DefaultHasher::new();
                for pos in self.iter().take(1000) {
                    pos.hash(&mut hasher);
                }
                hasher.finish()
            }
        }
    }

    /// Get the minimum item in this weight.
    pub fn min_item(&self) -> Option<usize> {
        match self {
            Self::Rs(rs) => rs.min_item(),
            Self::Bdd(bdd) => bdd.iter().next(),
            Self::BddBiodivine(bdd) => bdd.iter().next(),
            Self::Factored(fw) => fw.min_position(),
            Self::FactoredValidate(fw) => fw.rangeset().min_item(),
        }
    }

    /// Get the maximum item in this weight.
    pub fn max_item(&self) -> Option<usize> {
        match self {
            Self::Rs(rs) => rs.max_item(),
            Self::Bdd(bdd) => bdd.iter().last(),
            Self::BddBiodivine(bdd) => bdd.iter().last(),
            Self::Factored(fw) => fw.max_position(),
            Self::FactoredValidate(fw) => fw.rangeset().max_item(),
        }
    }

    /// Insert an item into this weight.
    pub fn insert(&mut self, item: usize) {
        match self {
            Self::Rs(rs) => rs.insert(item),
            Self::FactoredValidate(fw) => {
                let mut rs: RangeSet = fw.rangeset().clone();
                rs.insert(item);
                let dims = get_weight_dimensions();
                let factored = FactoredWeight::from_rsb(&rs.rsb, dims.num_tsids as usize);
                *self = Self::FactoredValidate(Arc::new(FactoredValidateWeight::new(factored, rs)));
            }
            Self::Bdd(_) | Self::BddBiodivine(_) | Self::Factored(_) => {
                // For BDD/Factored, convert to RangeSet, insert, convert back
                let mut rs: RangeSet = self.to_rangeset();
                rs.insert(item);
                *self = Self::Rs(rs);
            }
        }
    }

    /// Remove an item from this weight.
    pub fn remove(&mut self, item: usize) {
        match self {
            Self::Rs(rs) => rs.remove(item),
            Self::FactoredValidate(fw) => {
                let mut rs: RangeSet = fw.rangeset().clone();
                rs.remove(item);
                let dims = get_weight_dimensions();
                let factored = FactoredWeight::from_rsb(&rs.rsb, dims.num_tsids as usize);
                *self = Self::FactoredValidate(Arc::new(FactoredValidateWeight::new(factored, rs)));
            }
            Self::Bdd(_) | Self::BddBiodivine(_) | Self::Factored(_) => {
                // For BDD/Factored, convert to RangeSet, remove, convert back
                let mut rs: RangeSet = self.to_rangeset();
                rs.remove(item);
                *self = Self::Rs(rs);
            }
        }
    }

    /// Set an item to true (insert) or false (remove).
    pub fn set(&mut self, item: usize, value: bool) {
        if value {
            self.insert(item);
        } else {
            self.remove(item);
        }
    }

    /// Check if this weight is a subset of another.
    pub fn is_subset_of(&self, other: &Self) -> bool {
        // Fast path: empty is subset of anything
        if self.is_empty() {
            return true;
        }
        
        // Fast path: if other is "all", any weight is subset
        if other.is_all_fast() {
            return true;
        }
        
        match (self, other) {
            (Self::Rs(a), Self::Rs(b)) => a.is_subset_of(b),
            (Self::Bdd(a), Self::Bdd(b)) => a.subtract(b).is_empty(),
            (Self::Factored(a), Self::Factored(b)) => {
                // Use native 2D subset check
                a.is_subset_of(b)
            }
            (Self::FactoredValidate(a), Self::FactoredValidate(b)) => {
                a.rangeset().is_subset_of(b.rangeset())
            }
            _ => self.to_rangeset().is_subset_of(&other.to_rangeset()),
        }
    }

    /// Clip all values above a maximum, returning a new weight.
    pub fn clip_max(&self, max: usize) -> Self {
        match self {
            Self::Rs(rs) => {
                let mut clipped = rs.clone();
                clipped.clip_max(max);
                Self::Rs(clipped)
            }
            Self::Bdd(bdd) => {
                // For BDD, convert to ranges, filter, and rebuild
                let ranges: Vec<_> = bdd.to_rangeset().into_ranges()
                    .filter_map(|r| {
                        let start = *r.start();
                        let end = (*r.end()).min(max);
                        if start <= max { Some((start, end)) } else { None }
                    })
                    .collect();
                Self::from_ranges(&ranges)
            }
            Self::BddBiodivine(bdd) => {
                // For BDD, convert to ranges, filter, and rebuild
                let ranges: Vec<_> = bdd.to_rangeset().into_ranges()
                    .filter_map(|r| {
                        let start = *r.start();
                        let end = (*r.end()).min(max);
                        if start <= max { Some((start, end)) } else { None }
                    })
                    .collect();
                Self::from_ranges(&ranges)
            }
            Self::Factored(fw) => {
                // Use native 2D clip_max
                Self::Factored(Arc::new(fw.clip_max(max)))
            }
            Self::FactoredValidate(fw) => {
                let mut rs = fw.rangeset().clone();
                rs.clip_max(max);
                let dims = get_weight_dimensions();
                let factored = FactoredWeight::from_rsb(&rs.rsb, dims.num_tsids as usize);
                Self::FactoredValidate(Arc::new(FactoredValidateWeight::new(factored, rs)))
            }
        }
    }

    // ------------------------------------------------------------------------
    // Conversions
    // ------------------------------------------------------------------------

    /// Convert to RangeSet (for compatibility with existing code).
    /// NOTE: For Factored backend, this requires 1D expansion.
    pub fn to_rangeset(&self) -> RangeSet {
        match self {
            Self::Rs(rs) => rs.clone(),
            Self::Bdd(bdd) => RangeSet::from_rsb(bdd.to_rangeset()),
            Self::BddBiodivine(bdd) => RangeSet::from_rsb(bdd.to_rangeset()),
            Self::Factored(fw) => RangeSet::from_rsb(fw.expand_impl()),
            Self::FactoredValidate(fw) => fw.rangeset().clone(),
        }
    }

    /// Get the underlying RangeSetBlaze.
    /// NOTE: For Factored backend, this requires 1D expansion.
    pub fn to_rsb(&self) -> RangeSetBlaze<usize> {
        match self {
            Self::Rs(rs) => rs.rsb.clone(),
            Self::Bdd(bdd) => bdd.to_rangeset(),
            Self::BddBiodivine(bdd) => bdd.to_rangeset(),
            Self::Factored(fw) => fw.expand_impl(),
            Self::FactoredValidate(fw) => fw.rangeset().rsb.clone(),
        }
    }

    /// Get the underlying RangeSet reference (panics if BDD/Factored backend).
    pub fn as_rangeset(&self) -> &RangeSet {
        match self {
            Self::Rs(rs) => rs,
            Self::FactoredValidate(fw) => fw.rangeset(),
            Self::Bdd(_) | Self::BddBiodivine(_) | Self::Factored(_) => {
                panic!("Cannot get RangeSet reference from BDD/Factored weight")
            }
        }
    }

    // ------------------------------------------------------------------------
    // Iteration
    // ------------------------------------------------------------------------

    /// Iterate over positions in this weight.
    pub fn iter(&self) -> Box<dyn Iterator<Item = usize> + '_> {
        match self {
            Self::Rs(rs) => Box::new(rs.rsb.iter()),
            Self::Bdd(bdd) => Box::new(bdd.iter()),
            Self::BddBiodivine(bdd) => Box::new(bdd.iter()),
            Self::Factored(fw) => Box::new(fw.iter_positions()),
            Self::FactoredValidate(fw) => Box::new(fw.rangeset().rsb.iter()),
        }
    }

    /// Iterate over accepted positions up to a maximum value.
    pub fn iter_up_to(&self, max: usize) -> Box<dyn Iterator<Item = usize> + '_> {
        match self {
            Self::Rs(rs) => Box::new(rs.iter_up_to(max)),
            Self::Bdd(bdd) => Box::new(bdd.iter().take_while(move |&p| p <= max)),
            Self::BddBiodivine(bdd) => Box::new(bdd.iter().take_while(move |&p| p <= max)),
            Self::Factored(fw) => Box::new(fw.iter_positions_up_to(max)),
            Self::FactoredValidate(fw) => Box::new(fw.rangeset().iter_up_to(max)),
        }
    }

    /// Iterate over ranges.
    /// NOTE: For Factored backend, this requires 1D expansion.
    pub fn ranges(&self) -> Box<dyn Iterator<Item = std::ops::RangeInclusive<usize>> + '_> {
        match self {
            Self::Rs(rs) => Box::new(rs.rsb.ranges()),
            Self::Bdd(bdd) => Box::new(bdd.to_rangeset().into_ranges()),
            Self::BddBiodivine(bdd) => Box::new(bdd.to_rangeset().into_ranges()),
            Self::Factored(fw) => Box::new(fw.expand_impl().into_ranges()),
            Self::FactoredValidate(fw) => Box::new(fw.rangeset().rsb.ranges()),
        }
    }

    // ------------------------------------------------------------------------
    // Access to internal RangeSetBlaze (for compatibility)
    // ------------------------------------------------------------------------

    /// Access the internal RangeSetBlaze (panics if BDD/Factored backend).
    /// This is provided for compatibility with existing code that accesses `.rsb`.
    pub fn rsb(&self) -> &RangeSetBlaze<usize> {
        match self {
            Self::Rs(rs) => &rs.rsb,
            Self::FactoredValidate(fw) => &fw.rangeset().rsb,
            Self::Bdd(_) | Self::BddBiodivine(_) | Self::Factored(_) => {
                panic!("Cannot access rsb on BDD/Factored weight - use to_rsb() instead")
            }
        }
    }

    /// Get the number of ranges. Alias for `num_ranges()` for compatibility with RangeSetBlaze.
    pub fn ranges_len(&self) -> usize {
        self.num_ranges()
    }

    /// Get the last (maximum) element, if any. Alias for `max_item()` for compatibility.
    pub fn last(&self) -> Option<usize> {
        self.max_item()
    }

    /// Get the first (minimum) element, if any. Alias for `min_item()` for compatibility.
    pub fn first(&self) -> Option<usize> {
        self.min_item()
    }

    /// Get a unique identifier for this weight's underlying data.
    /// Used for deduplication of interned weights.
    /// For RangeSet backend, this is the Arc pointer address.
    /// For BDD backend, this is also the Arc pointer address.
    pub fn intern_id(&self) -> usize {
        match self {
            Self::Rs(rs) => rs.intern_id(),
            Self::Bdd(bdd) => Arc::as_ptr(bdd) as usize,
            Self::BddBiodivine(bdd) => Arc::as_ptr(bdd) as usize,
            Self::Factored(fw) => Arc::as_ptr(fw) as usize,
            Self::FactoredValidate(fw) => Arc::as_ptr(fw) as usize,
        }
    }
}

// ============================================================================
// Operator Implementations
// ============================================================================

impl BitOr for &AbstractWeight {
    type Output = AbstractWeight;

    fn bitor(self, rhs: Self) -> Self::Output {
        // Fast path: if either is empty, return the other
        if self.is_empty() {
            return rhs.clone();
        }
        if rhs.is_empty() {
            return self.clone();
        }
        
        // Fast path: if either is "all", return "all"
        if self.is_all_fast() {
            return self.clone();
        }
        if rhs.is_all_fast() {
            return rhs.clone();
        }
        
        match (self, rhs) {
            (AbstractWeight::Rs(a), AbstractWeight::Rs(b)) => AbstractWeight::Rs(a | b),
            (AbstractWeight::Bdd(a), AbstractWeight::Bdd(b)) => {
                AbstractWeight::Bdd(Arc::new(a.union(b)))
            }
            (AbstractWeight::BddBiodivine(a), AbstractWeight::BddBiodivine(b)) => {
                AbstractWeight::BddBiodivine(Arc::new(a.union(b)))
            }
            (AbstractWeight::Factored(a), AbstractWeight::Factored(b)) => {
                AbstractWeight::Factored(Arc::new(a.union(b)))
            }
            (AbstractWeight::FactoredValidate(a), AbstractWeight::FactoredValidate(b)) => {
                let factored = a.factored().union(b.factored());
                let rangeset = a.rangeset() | b.rangeset();
                AbstractWeight::FactoredValidate(Arc::new(FactoredValidateWeight::new(factored, rangeset)))
            }
            _ => {
                // Cross-backend: convert to common format
                let a_rs = self.to_rangeset();
                let b_rs = rhs.to_rangeset();
                AbstractWeight::Rs(&a_rs | &b_rs)
            }
        }
    }
}

impl BitOr for AbstractWeight {
    type Output = AbstractWeight;

    fn bitor(self, rhs: Self) -> Self::Output {
        &self | &rhs
    }
}

impl BitOrAssign<&AbstractWeight> for AbstractWeight {
    fn bitor_assign(&mut self, rhs: &Self) {
        *self = &*self | rhs;
    }
}

impl BitAnd for &AbstractWeight {
    type Output = AbstractWeight;

    fn bitand(self, rhs: Self) -> Self::Output {
        // Fast path: if either is empty, return empty
        if self.is_empty() || rhs.is_empty() {
            return AbstractWeight::zeros();
        }
        
        // Fast path: if one is "all", return the other
        if self.is_all_fast() {
            return rhs.clone();
        }
        if rhs.is_all_fast() {
            return self.clone();
        }
        
        match (self, rhs) {
            (AbstractWeight::Rs(a), AbstractWeight::Rs(b)) => AbstractWeight::Rs(a & b),
            (AbstractWeight::Bdd(a), AbstractWeight::Bdd(b)) => {
                AbstractWeight::Bdd(Arc::new(a.intersection(b)))
            }
            (AbstractWeight::BddBiodivine(a), AbstractWeight::BddBiodivine(b)) => {
                AbstractWeight::BddBiodivine(Arc::new(a.intersection(b)))
            }
            (AbstractWeight::Factored(a), AbstractWeight::Factored(b)) => {
                AbstractWeight::Factored(Arc::new(a.intersection(b)))
            }
            (AbstractWeight::FactoredValidate(a), AbstractWeight::FactoredValidate(b)) => {
                let factored = a.factored().intersection(b.factored());
                let rangeset = a.rangeset() & b.rangeset();
                AbstractWeight::FactoredValidate(Arc::new(FactoredValidateWeight::new(factored, rangeset)))
            }
            _ => {
                let a_rs = self.to_rangeset();
                let b_rs = rhs.to_rangeset();
                AbstractWeight::Rs(&a_rs & &b_rs)
            }
        }
    }
}

impl BitAnd for AbstractWeight {
    type Output = AbstractWeight;

    fn bitand(self, rhs: Self) -> Self::Output {
        &self & &rhs
    }
}

impl BitAndAssign<&AbstractWeight> for AbstractWeight {
    fn bitand_assign(&mut self, rhs: &Self) {
        *self = &*self & rhs;
    }
}

impl Not for &AbstractWeight {
    type Output = AbstractWeight;

    fn not(self) -> Self::Output {
        match self {
            AbstractWeight::Rs(rs) => AbstractWeight::Rs(!rs),
            AbstractWeight::Bdd(bdd) => AbstractWeight::Bdd(Arc::new(bdd.complement())),
            AbstractWeight::BddBiodivine(bdd) => AbstractWeight::BddBiodivine(Arc::new(bdd.complement())),
            AbstractWeight::Factored(_) => {
                // FactoredWeight doesn't support complement, fall back to RangeSet
                AbstractWeight::Rs(!&self.to_rangeset())
            }
            AbstractWeight::FactoredValidate(fw) => {
                let rangeset = !fw.rangeset();
                let dims = get_weight_dimensions();
                let factored = FactoredWeight::from_rsb(&rangeset.rsb, dims.num_tsids as usize);
                AbstractWeight::FactoredValidate(Arc::new(FactoredValidateWeight::new(factored, rangeset)))
            }
        }
    }
}

impl Not for AbstractWeight {
    type Output = AbstractWeight;

    fn not(self) -> Self::Output {
        !&self
    }
}

impl Sub for &AbstractWeight {
    type Output = AbstractWeight;

    fn sub(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (AbstractWeight::Rs(a), AbstractWeight::Rs(b)) => AbstractWeight::Rs(a - b),
            (AbstractWeight::Bdd(a), AbstractWeight::Bdd(b)) => {
                AbstractWeight::Bdd(Arc::new(a.subtract(b)))
            }
            (AbstractWeight::BddBiodivine(a), AbstractWeight::BddBiodivine(b)) => {
                AbstractWeight::BddBiodivine(Arc::new(a.subtract(b)))
            }
            (AbstractWeight::Factored(_), AbstractWeight::Factored(_)) => {
                // FactoredWeight doesn't support subtract, fall back to RangeSet
                let a_rs = self.to_rangeset();
                let b_rs = rhs.to_rangeset();
                AbstractWeight::Rs(&a_rs - &b_rs)
            }
            (AbstractWeight::FactoredValidate(a), AbstractWeight::FactoredValidate(b)) => {
                let rangeset = a.rangeset() - b.rangeset();
                let dims = get_weight_dimensions();
                let factored = FactoredWeight::from_rsb(&rangeset.rsb, dims.num_tsids as usize);
                AbstractWeight::FactoredValidate(Arc::new(FactoredValidateWeight::new(factored, rangeset)))
            }
            _ => {
                let a_rs = self.to_rangeset();
                let b_rs = rhs.to_rangeset();
                AbstractWeight::Rs(&a_rs - &b_rs)
            }
        }
    }
}

impl Sub for AbstractWeight {
    type Output = AbstractWeight;

    fn sub(self, rhs: Self) -> Self::Output {
        &self - &rhs
    }
}

impl SubAssign<&AbstractWeight> for AbstractWeight {
    fn sub_assign(&mut self, rhs: &Self) {
        *self = &*self - rhs;
    }
}

// ============================================================================
// Trait Implementations
// ============================================================================

impl Default for AbstractWeight {
    fn default() -> Self {
        Self::zeros()
    }
}

impl PartialEq for AbstractWeight {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Rs(a), Self::Rs(b)) => a == b,
            (Self::Bdd(a), Self::Bdd(b)) => a == b,
            _ => self.to_rsb() == other.to_rsb(),
        }
    }
}

impl Eq for AbstractWeight {}

// Arbitrary ordering for use in semiring operations (e.g. BitsetWeight).
// This is a lexicographic comparison: compare by length, then by elements.
impl PartialOrd for AbstractWeight {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for AbstractWeight {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        // First compare by length
        match self.len().cmp(&other.len()) {
            Ordering::Equal => {}
            other => return other,
        }
        // Then compare element-by-element (lexicographically)
        let mut self_iter = self.iter();
        let mut other_iter = other.iter();
        loop {
            match (self_iter.next(), other_iter.next()) {
                (Some(a), Some(b)) => {
                    match a.cmp(&b) {
                        Ordering::Equal => continue,
                        other => return other,
                    }
                }
                (None, None) => return Ordering::Equal,
                (Some(_), None) => return Ordering::Greater,
                (None, Some(_)) => return Ordering::Less,
            }
        }
    }
}

impl Hash for AbstractWeight {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            Self::Rs(rs) => rs.hash(state),
            Self::Bdd(bdd) => {
                // Hash based on sorted positions for consistency
                for pos in bdd.iter() {
                    pos.hash(state);
                }
            }
            Self::BddBiodivine(bdd) => {
                // Hash based on sorted positions for consistency
                for pos in bdd.iter() {
                    pos.hash(state);
                }
            }
            Self::Factored(fw) => {
                // Hash based on 2D structure for consistency
                fw.hash_2d(state);
            }
            Self::FactoredValidate(fw) => {
                fw.rangeset().hash(state);
            }
        }
    }
}

impl Debug for AbstractWeight {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Rs(rs) => write!(f, "Weight::Rs({:?})", rs),
            Self::Bdd(bdd) => write!(f, "Weight::Bdd({} nodes)", bdd.num_nodes()),
            Self::BddBiodivine(bdd) => write!(f, "Weight::BddBiodivine({} nodes)", bdd.num_nodes()),
            Self::Factored(fw) => write!(f, "Weight::Factored({} terms)", fw.num_terms()),
        }
    }
}

impl Display for AbstractWeight {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Rs(rs) => write!(f, "{}", rs),
            Self::Bdd(bdd) => {
                let ranges: Vec<_> = bdd.to_rangeset().into_ranges().collect();
                if ranges.is_empty() {
                    write!(f, "∅")
                } else if ranges.len() <= 5 {
                    let strs: Vec<String> = ranges.iter()
                        .map(|r| format!("{}..={}", r.start(), r.end()))
                        .collect();
                    write!(f, "[{}]", strs.join(", "))
                } else {
                    write!(f, "[{} ranges]", ranges.len())
                }
            }
            Self::BddBiodivine(bdd) => {
                let ranges: Vec<_> = bdd.to_rangeset().into_ranges().collect();
                if ranges.is_empty() {
                    write!(f, "∅")
                } else if ranges.len() <= 5 {
                    let strs: Vec<String> = ranges.iter()
                        .map(|r| format!("{}..={}", r.start(), r.end()))
                        .collect();
                    write!(f, "[{}]", strs.join(", "))
                } else {
                    write!(f, "[{} ranges]", ranges.len())
                }
            }
            Self::Factored(fw) => {
                let num_terms = fw.num_terms();
                if fw.is_empty() {
                    write!(f, "∅")
                } else if num_terms <= 3 {
                    write!(f, "[Factored: {} terms]", num_terms)
                } else {
                    write!(f, "[Factored: {} terms]", num_terms)
                }
            }
            Self::FactoredValidate(fw) => {
                let num_terms = fw.factored().num_terms();
                if fw.rangeset().is_empty() {
                    write!(f, "∅")
                } else {
                    write!(f, "[FactoredValidate: {} terms, {} ranges]", num_terms, fw.rangeset().num_ranges())
                }
            }
        }
    }
}

impl FromIterator<usize> for AbstractWeight {
    fn from_iter<I: IntoIterator<Item = usize>>(iter: I) -> Self {
        match get_weight_backend() {
            WeightBackend::RangeSet => Self::Rs(RangeSet::from_iter(iter)),
            WeightBackend::Bdd | WeightBackend::BddBiodivine | WeightBackend::Factored | WeightBackend::FactoredValidate => {
                let rsb: RangeSetBlaze<usize> = iter.into_iter().collect();
                Self::from_rsb(rsb)
            }
        }
    }
}

impl<'a> FromIterator<&'a usize> for AbstractWeight {
    fn from_iter<I: IntoIterator<Item = &'a usize>>(iter: I) -> Self {
        iter.into_iter().copied().collect()
    }
}

// FromIterator for ranges
impl FromIterator<std::ops::RangeInclusive<usize>> for AbstractWeight {
    fn from_iter<I: IntoIterator<Item = std::ops::RangeInclusive<usize>>>(iter: I) -> Self {
        match get_weight_backend() {
            WeightBackend::RangeSet => Self::Rs(RangeSet::from_iter(iter)),
            WeightBackend::Bdd | WeightBackend::BddBiodivine | WeightBackend::Factored | WeightBackend::FactoredValidate => {
                let ranges: Vec<_> = iter.into_iter().map(|r| (*r.start(), *r.end())).collect();
                Self::from_ranges(&ranges)
            }
        }
    }
}

// ============================================================================
// Serialization
// ============================================================================

impl Serialize for AbstractWeight {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Always serialize as RangeSet for compatibility
        self.to_rangeset().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for AbstractWeight {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        // Deserialize as RangeSet, then convert based on backend
        let rs = RangeSet::deserialize(deserializer)?;
        match get_weight_backend() {
            WeightBackend::RangeSet => Ok(Self::Rs(rs)),
            WeightBackend::Bdd => {
                let dims = get_weight_dimensions();
                let ranges: Vec<_> = rs.rsb.ranges().map(|r| (*r.start(), *r.end())).collect();
                Ok(Self::Bdd(Arc::new(BddWeight::from_ranges(
                    ranges.into_iter(),
                    dims.num_tsids as u16,
                    dims.num_tokens as u16,
                ))))
            }
            WeightBackend::BddBiodivine => {
                let dims = get_weight_dimensions();
                let ranges: Vec<_> = rs.rsb.ranges().map(|r| (*r.start(), *r.end())).collect();
                Ok(Self::BddBiodivine(Arc::new(BddWeightBiodivine::from_ranges(
                    ranges.into_iter(),
                    dims.num_tsids as u16,
                    dims.num_tokens as u16,
                ))))
            }
            WeightBackend::Factored => {
                let dims = get_weight_dimensions();
                let ranges: Vec<_> = rs.rsb.ranges().map(|r| (*r.start(), *r.end())).collect();
                Ok(Self::Factored(Arc::new(FactoredWeight::from_1d_ranges(
                    ranges.into_iter(),
                    dims.num_tsids,
                ))))
            }
            WeightBackend::FactoredValidate => {
                let dims = get_weight_dimensions();
                let ranges: Vec<_> = rs.rsb.ranges().map(|r| (*r.start(), *r.end())).collect();
                let factored = FactoredWeight::from_1d_ranges(ranges.into_iter(), dims.num_tsids);
                Ok(Self::FactoredValidate(Arc::new(FactoredValidateWeight::new(factored, rs))))
            }
        }
    }
}

// ============================================================================
// JSONConvertible
// ============================================================================

use crate::json_serialization::{JSONConvertible, JSONNode};

impl JSONConvertible for AbstractWeight {
    fn to_json(&self) -> JSONNode {
        // Delegate to RangeSet implementation
        self.to_rangeset().to_json()
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        // Delegate to RangeSet implementation, then convert to appropriate backend
        let rs = RangeSet::from_json(node)?;
        Ok(Self::from(rs))
    }
}

// ============================================================================
// Conversion From/Into RangeSet
// ============================================================================

impl From<RangeSet> for AbstractWeight {
    fn from(rs: RangeSet) -> Self {
        match get_weight_backend() {
            WeightBackend::RangeSet => Self::Rs(rs),
            WeightBackend::Bdd => {
                let dims = get_weight_dimensions();
                let ranges: Vec<_> = rs.rsb.ranges().map(|r| (*r.start(), *r.end())).collect();
                Self::Bdd(Arc::new(BddWeight::from_ranges(
                    ranges.into_iter(),
                    dims.num_tsids as u16,
                    dims.num_tokens as u16,
                )))
            }
            WeightBackend::BddBiodivine => {
                let dims = get_weight_dimensions();
                let ranges: Vec<_> = rs.rsb.ranges().map(|r| (*r.start(), *r.end())).collect();
                Self::BddBiodivine(Arc::new(BddWeightBiodivine::from_ranges(
                    ranges.into_iter(),
                    dims.num_tsids as u16,
                    dims.num_tokens as u16,
                )))
            }
            WeightBackend::Factored => {
                let dims = get_weight_dimensions();
                let ranges: Vec<_> = rs.rsb.ranges().map(|r| (*r.start(), *r.end())).collect();
                Self::Factored(Arc::new(FactoredWeight::from_1d_ranges(
                    ranges.into_iter(),
                    dims.num_tsids,
                )))
            }
            WeightBackend::FactoredValidate => {
                let dims = get_weight_dimensions();
                let ranges: Vec<_> = rs.rsb.ranges().map(|r| (*r.start(), *r.end())).collect();
                let factored = FactoredWeight::from_1d_ranges(ranges.into_iter(), dims.num_tsids);
                Self::FactoredValidate(Arc::new(FactoredValidateWeight::new(factored, rs)))
            }
        }
    }
}

impl From<AbstractWeight> for RangeSet {
    fn from(w: AbstractWeight) -> Self {
        w.to_rangeset()
    }
}

// Conversion from hybrid_bitset::RangeSet (the Arc<RangeSetBlaze> type)
impl From<crate::datastructures::hybrid_bitset::RangeSet> for AbstractWeight {
    fn from(hbrs: crate::datastructures::hybrid_bitset::RangeSet) -> Self {
        let rsb: RangeSetBlaze<usize> = hbrs.inner.as_ref().clone();
        Self::from_rsb(rsb)
    }
}

// Conversion from AbstractWeight to hybrid_bitset::RangeSet
impl From<AbstractWeight> for crate::datastructures::hybrid_bitset::RangeSet {
    fn from(w: AbstractWeight) -> Self {
        Self::from(w.to_rsb())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros_and_all() {
        let z = AbstractWeight::zeros();
        assert!(z.is_empty());
        assert!(!z.is_all_fast());

        let a = AbstractWeight::all();
        assert!(!a.is_empty());
        assert!(a.is_all_fast());
    }

    #[test]
    fn test_from_item() {
        let w = AbstractWeight::from_item(42);
        assert!(w.contains(42));
        assert!(!w.contains(41));
        assert!(!w.contains(43));
    }

    #[test]
    fn test_from_ranges() {
        let w = AbstractWeight::from_ranges(&[(0, 10), (20, 30)]);
        assert!(w.contains(5));
        assert!(w.contains(25));
        assert!(!w.contains(15));
    }

    #[test]
    fn test_union() {
        let a = AbstractWeight::from_ranges(&[(0, 5)]);
        let b = AbstractWeight::from_ranges(&[(10, 15)]);
        let c = &a | &b;

        assert!(c.contains(3));
        assert!(c.contains(12));
        assert!(!c.contains(7));
    }

    #[test]
    fn test_intersection() {
        let a = AbstractWeight::from_ranges(&[(0, 10)]);
        let b = AbstractWeight::from_ranges(&[(5, 15)]);
        let c = &a & &b;

        assert!(c.contains(7));
        assert!(!c.contains(3));
        assert!(!c.contains(12));
    }

    #[test]
    fn test_subtraction() {
        let a = AbstractWeight::from_ranges(&[(0, 10)]);
        let b = AbstractWeight::from_ranges(&[(5, 15)]);
        let c = &a - &b;

        assert!(c.contains(3));
        assert!(!c.contains(7));
    }

    #[test]
    fn test_complement() {
        let a = AbstractWeight::from_ranges(&[(5, 10)]);
        let b = !&a;

        assert!(!b.contains(7));
        assert!(b.contains(3));
        assert!(b.contains(15));
    }

    #[test]
    fn test_equality() {
        let a = AbstractWeight::from_ranges(&[(0, 10)]);
        let b = AbstractWeight::from_ranges(&[(0, 10)]);
        let c = AbstractWeight::from_ranges(&[(0, 11)]);

        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_from_iter() {
        let w: AbstractWeight = vec![1, 3, 5, 7].into_iter().collect();
        assert!(w.contains(1));
        assert!(w.contains(5));
        assert!(!w.contains(2));
    }

    #[test]
    fn test_to_rangeset_roundtrip() {
        let original = AbstractWeight::from_ranges(&[(5, 15), (25, 35)]);
        let rs = original.to_rangeset();
        let back = AbstractWeight::from(rs);

        assert_eq!(original, back);
    }

    #[test]
    fn test_factored_complement_via_rangeset() {
        use super::set_weight_dimensions;
        
        // Set small dimensions for testing
        let dims = WeightDimensions::new(10, 5);  // 10 tokens, 5 tsids
        set_weight_dimensions(dims);
        
        // Create a factored weight: all tokens × tsid 0
        let fw = FactoredWeight::from_product(
            (0..10u16).collect(),  // all tokens
            std::iter::once(0u16).collect(),  // just tsid 0
            5,  // num_tsids
        );
        let w = AbstractWeight::Factored(Arc::new(fw));
        
        // Check that it contains expected positions
        assert!(w.contains(0));   // token 0, tsid 0 -> pos 0
        assert!(w.contains(5));   // token 1, tsid 0 -> pos 5
        assert!(!w.contains(1));  // token 0, tsid 1 -> pos 1
        
        // Get the complement (which converts to RangeSet)
        let w_not = !&w;
        
        // The complement should contain tsids 1-4 for all tokens
        assert!(!w_not.contains(0));   // token 0, tsid 0 -> NOT in complement
        assert!(!w_not.contains(5));   // token 1, tsid 0 -> NOT in complement
        assert!(w_not.contains(1));    // token 0, tsid 1 -> IN complement
        assert!(w_not.contains(2));    // token 0, tsid 2 -> IN complement
        
        // Union should be all()
        let union = &w | &w_not;
        
        // Check that all positions are in union
        for token in 0..10 {
            for tsid in 0..5 {
                let pos = token * 5 + tsid;
                assert!(union.contains(pos), "pos {} should be in union", pos);
            }
        }
    }
}
