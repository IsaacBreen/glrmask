//! Abstract weight type that can be backed by different storage formats.
//!
//! This module provides an `AbstractWeight` enum that wraps either:
//! - `RangeSet`: Sparse range-based storage (default, fast operations, good for sparse weights)
//! - `BddWeight`: Binary Decision Diagram storage (custom 5-byte nodes, memory efficient)
//! - `BddWeightBiodivine`: BDD storage using biodivine_lib_bdd (battle-tested, fewer nodes)
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
pub fn get_weight_backend() -> WeightBackend {
    static BACKEND: OnceLock<WeightBackend> = OnceLock::new();
    *BACKEND.get_or_init(|| {
        match std::env::var("WEIGHT_BACKEND").as_deref() {
            Ok("bdd") | Ok("BDD") => WeightBackend::Bdd,
            Ok("bdd-biodivine") | Ok("BDD-BIODIVINE") | Ok("biodivine") => WeightBackend::BddBiodivine,
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
/// This is a thin wrapper around either `RangeSet` (default), `BddWeight`, or `BddWeightBiodivine`.
/// All operations delegate to the underlying implementation.
#[derive(Clone)]
pub enum AbstractWeight {
    /// RangeSet backend (default, interned).
    Rs(RangeSet),
    /// Custom BddWeight backend (5-byte nodes, memory efficient).
    Bdd(Arc<BddWeight>),
    /// Biodivine BddWeight backend (battle-tested, fewer nodes).
    BddBiodivine(Arc<BddWeightBiodivine>),
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
        }
    }

    /// Fast check if this weight is "all" (accepts everything).
    pub fn is_all_fast(&self) -> bool {
        match self {
            Self::Rs(rs) => rs.is_all_fast(),
            Self::Bdd(bdd) => bdd.is_full(),
            Self::BddBiodivine(bdd) => bdd.is_full(),
        }
    }

    /// Check if a position is contained in this weight.
    pub fn contains(&self, pos: usize) -> bool {
        match self {
            Self::Rs(rs) => rs.contains(pos),
            Self::Bdd(bdd) => bdd.contains_pos(pos),
            Self::BddBiodivine(bdd) => bdd.contains_pos(pos),
        }
    }

    /// Check if two weights are disjoint.
    pub fn is_disjoint(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Rs(a), Self::Rs(b)) => a.is_disjoint(b),
            (Self::Bdd(a), Self::Bdd(b)) => a.intersection(b).is_empty(),
            _ => self.to_rangeset().is_disjoint(&other.to_rangeset()),
        }
    }

    /// Get the number of ranges (only meaningful for RangeSet backend).
    pub fn num_ranges(&self) -> usize {
        match self {
            Self::Rs(rs) => rs.num_ranges(),
            Self::Bdd(bdd) => bdd.to_rangeset().ranges_len(),
            Self::BddBiodivine(bdd) => bdd.to_rangeset().ranges_len(),
        }
    }

    /// Get the number of elements in this weight.
    pub fn len(&self) -> usize {
        match self {
            Self::Rs(rs) => rs.len(),
            Self::Bdd(bdd) => bdd.len(),
            Self::BddBiodivine(bdd) => bdd.len(),
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
            Self::Bdd(_) | Self::BddBiodivine(_) => {
                // For BDD, compute a hash of the ranges
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
        }
    }

    /// Get the maximum item in this weight.
    pub fn max_item(&self) -> Option<usize> {
        match self {
            Self::Rs(rs) => rs.max_item(),
            Self::Bdd(bdd) => bdd.iter().last(),
            Self::BddBiodivine(bdd) => bdd.iter().last(),
        }
    }

    /// Insert an item into this weight.
    pub fn insert(&mut self, item: usize) {
        match self {
            Self::Rs(rs) => rs.insert(item),
            Self::Bdd(_) | Self::BddBiodivine(_) => {
                // For BDD, convert to RangeSet, insert, convert back
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
            Self::Bdd(_) | Self::BddBiodivine(_) => {
                // For BDD, convert to RangeSet, remove, convert back
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
        match (self, other) {
            (Self::Rs(a), Self::Rs(b)) => a.is_subset_of(b),
            (Self::Bdd(a), Self::Bdd(b)) => a.subtract(b).is_empty(),
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
        }
    }

    // ------------------------------------------------------------------------
    // Conversions
    // ------------------------------------------------------------------------

    /// Convert to RangeSet (for compatibility with existing code).
    pub fn to_rangeset(&self) -> RangeSet {
        match self {
            Self::Rs(rs) => rs.clone(),
            Self::Bdd(bdd) => RangeSet::from_rsb(bdd.to_rangeset()),
            Self::BddBiodivine(bdd) => RangeSet::from_rsb(bdd.to_rangeset()),
        }
    }

    /// Get the underlying RangeSetBlaze.
    pub fn to_rsb(&self) -> RangeSetBlaze<usize> {
        match self {
            Self::Rs(rs) => rs.rsb.clone(),
            Self::Bdd(bdd) => bdd.to_rangeset(),
            Self::BddBiodivine(bdd) => bdd.to_rangeset(),
        }
    }

    /// Get the underlying RangeSet reference (panics if BDD backend).
    pub fn as_rangeset(&self) -> &RangeSet {
        match self {
            Self::Rs(rs) => rs,
            Self::Bdd(_) | Self::BddBiodivine(_) => panic!("Cannot get RangeSet reference from BDD weight"),
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
        }
    }

    /// Iterate over accepted positions up to a maximum value.
    pub fn iter_up_to(&self, max: usize) -> Box<dyn Iterator<Item = usize> + '_> {
        match self {
            Self::Rs(rs) => Box::new(rs.iter_up_to(max)),
            Self::Bdd(bdd) => Box::new(bdd.iter().take_while(move |&p| p <= max)),
            Self::BddBiodivine(bdd) => Box::new(bdd.iter().take_while(move |&p| p <= max)),
        }
    }

    /// Iterate over ranges.
    pub fn ranges(&self) -> Box<dyn Iterator<Item = std::ops::RangeInclusive<usize>> + '_> {
        match self {
            Self::Rs(rs) => Box::new(rs.rsb.ranges()),
            Self::Bdd(bdd) => Box::new(bdd.to_rangeset().into_ranges()),
            Self::BddBiodivine(bdd) => Box::new(bdd.to_rangeset().into_ranges()),
        }
    }

    // ------------------------------------------------------------------------
    // Access to internal RangeSetBlaze (for compatibility)
    // ------------------------------------------------------------------------

    /// Access the internal RangeSetBlaze (panics if BDD backend).
    /// This is provided for compatibility with existing code that accesses `.rsb`.
    pub fn rsb(&self) -> &RangeSetBlaze<usize> {
        match self {
            Self::Rs(rs) => &rs.rsb,
            Self::Bdd(_) | Self::BddBiodivine(_) => panic!("Cannot access rsb on BDD weight - use to_rsb() instead"),
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
        }
    }
}

// ============================================================================
// Operator Implementations
// ============================================================================

impl BitOr for &AbstractWeight {
    type Output = AbstractWeight;

    fn bitor(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (AbstractWeight::Rs(a), AbstractWeight::Rs(b)) => AbstractWeight::Rs(a | b),
            (AbstractWeight::Bdd(a), AbstractWeight::Bdd(b)) => {
                AbstractWeight::Bdd(Arc::new(a.union(b)))
            }
            (AbstractWeight::BddBiodivine(a), AbstractWeight::BddBiodivine(b)) => {
                AbstractWeight::BddBiodivine(Arc::new(a.union(b)))
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
        match (self, rhs) {
            (AbstractWeight::Rs(a), AbstractWeight::Rs(b)) => AbstractWeight::Rs(a & b),
            (AbstractWeight::Bdd(a), AbstractWeight::Bdd(b)) => {
                AbstractWeight::Bdd(Arc::new(a.intersection(b)))
            }
            (AbstractWeight::BddBiodivine(a), AbstractWeight::BddBiodivine(b)) => {
                AbstractWeight::BddBiodivine(Arc::new(a.intersection(b)))
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
        }
    }
}

impl Debug for AbstractWeight {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Rs(rs) => write!(f, "Weight::Rs({:?})", rs),
            Self::Bdd(bdd) => write!(f, "Weight::Bdd({} nodes)", bdd.num_nodes()),
            Self::BddBiodivine(bdd) => write!(f, "Weight::BddBiodivine({} nodes)", bdd.num_nodes()),
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
        }
    }
}

impl FromIterator<usize> for AbstractWeight {
    fn from_iter<I: IntoIterator<Item = usize>>(iter: I) -> Self {
        match get_weight_backend() {
            WeightBackend::RangeSet => Self::Rs(RangeSet::from_iter(iter)),
            WeightBackend::Bdd | WeightBackend::BddBiodivine => {
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
            WeightBackend::Bdd | WeightBackend::BddBiodivine => {
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
}
