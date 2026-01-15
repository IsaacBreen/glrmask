//! Abstract weight type that can be backed by different storage formats.
//!
//! This module provides an `AbstractWeight` enum that wraps either:
//! - `RangeSet`: Sparse range-based storage (default, fast operations, good for sparse weights)
//! - `BddWeight`: Binary Decision Diagram storage (biodivine-backed)
//! - `BddWeightBiodivine`: BDD storage using biodivine_lib_bdd (kept for compatibility/experiments)
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
//! - `WEIGHT_BACKEND=bdd`: Uses `BddWeight` (biodivine-backed)
//! - `WEIGHT_BACKEND=bdd-biodivine`: Uses `BddWeightBiodivine` (biodivine_lib_bdd)
//! - `WEIGHT_BACKEND=factored`: Uses 2D factored representation (experimental)
//! - `WEIGHT_BACKEND=factored-validate`: Uses factored + RangeSet with validation

use std::fmt::{Debug, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, Sub, SubAssign};
use std::sync::{Arc, OnceLock};
use std::iter::FromIterator;

use range_set_blaze::RangeSetBlaze;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::rangeset::{RangeSet, RangeSetInner};
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
    /// Binary Decision Diagram storage.
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
/// - `bdd`: BddWeight (biodivine-backed)
/// - `bdd-biodivine`: biodivine_lib_bdd (BddWeightBiodivine)
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
    /// BddWeight backend.
    Bdd(Arc<BddWeight>),
    /// Biodivine BddWeight backend (battle-tested, fewer nodes).
    BddBiodivine(Arc<BddWeightBiodivine>),
    /// 2D factored representation (union of Cartesian products).
    Factored(Arc<FactoredWeight>),
    /// Factored + RangeSet with validation.
    FactoredValidate(Arc<FactoredValidateWeight>),
}

// ============================================================================
// Backend trait + dispatch macros
// ============================================================================

trait BackendOps: Sized {
    // Constructors
    fn zeros() -> Self;
    fn from_item(item: usize) -> Self;
    fn from_ranges(ranges: &[(usize, usize)]) -> Self;
    fn from_rsb(rsb: RangeSetBlaze<usize>) -> Self;
    fn ones(len: usize) -> Self;
    fn from_token_set_all_tsids(tokens: RangeSetBlaze<usize>, dims: WeightDimensions) -> Self;
    fn from_token_set_specific_tsid(tokens: RangeSetBlaze<usize>, tsid: usize, dims: WeightDimensions) -> Self;
    fn tsid_columns_with_dims(tsids: Vec<usize>, num_tsids: usize, num_tokens: usize) -> Self;

    // Queries
    fn num_ranges(&self) -> usize;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn is_all_fast(&self) -> bool;
    fn contains(&self, pos: usize) -> bool;
    fn is_disjoint(&self, other: &Self) -> bool;
    fn is_subset_of(&self, other: &Self) -> bool;
    fn min_item(&self) -> Option<usize>;
    fn max_item(&self) -> Option<usize>;
    fn fp(&self) -> u64;
    fn intern_id(&self) -> usize;

    // Iteration / conversion
    fn iter<'a>(&'a self) -> Box<dyn Iterator<Item = usize> + 'a>;
    fn iter_up_to<'a>(&'a self, max: usize) -> Box<dyn Iterator<Item = usize> + 'a>;
    fn ranges<'a>(&'a self) -> Box<dyn Iterator<Item = std::ops::RangeInclusive<usize>> + 'a>;
    fn to_rangeset(&self) -> RangeSet;
    fn to_rsb(&self) -> RangeSetBlaze<usize>;

    // Operations
    fn union(&self, other: &Self) -> Self;
    fn intersection(&self, other: &Self) -> Self;
    fn subtract(&self, other: &Self) -> Self;
    fn clip_max(&self, max: usize) -> Self;
    fn insert(&mut self, item: usize);
    fn remove(&mut self, item: usize);

    // Formatting
    fn fmt_display(&self, f: &mut Formatter<'_>) -> std::fmt::Result;
}

fn wrap_rs(v: RangeSet) -> AbstractWeight { AbstractWeight::Rs(v) }
fn wrap_bdd(v: Arc<BddWeight>) -> AbstractWeight { AbstractWeight::Bdd(v) }
fn wrap_bdd_biodivine(v: Arc<BddWeightBiodivine>) -> AbstractWeight { AbstractWeight::BddBiodivine(v) }
fn wrap_factored(v: Arc<FactoredWeight>) -> AbstractWeight { AbstractWeight::Factored(v) }
fn wrap_factored_validate(v: Arc<FactoredValidateWeight>) -> AbstractWeight { AbstractWeight::FactoredValidate(v) }

macro_rules! dispatch_ref {
    ($self:expr, $method:ident $(, $args:expr)*) => {{
        match $self {
            AbstractWeight::Rs(inner) => <RangeSet as BackendOps>::$method(inner $(, $args)*),
            AbstractWeight::Bdd(inner) => <Arc<BddWeight> as BackendOps>::$method(inner $(, $args)*),
            AbstractWeight::BddBiodivine(inner) => <Arc<BddWeightBiodivine> as BackendOps>::$method(inner $(, $args)*),
            AbstractWeight::Factored(inner) => <Arc<FactoredWeight> as BackendOps>::$method(inner $(, $args)*),
            AbstractWeight::FactoredValidate(inner) => <Arc<FactoredValidateWeight> as BackendOps>::$method(inner $(, $args)*),
        }
    }};
}

macro_rules! dispatch_ref_to_weight {
    ($self:expr, $method:ident $(, $args:expr)*) => {{
        match $self {
            AbstractWeight::Rs(inner) => wrap_rs(<RangeSet as BackendOps>::$method(inner $(, $args)*)),
            AbstractWeight::Bdd(inner) => wrap_bdd(<Arc<BddWeight> as BackendOps>::$method(inner $(, $args)*)),
            AbstractWeight::BddBiodivine(inner) => wrap_bdd_biodivine(<Arc<BddWeightBiodivine> as BackendOps>::$method(inner $(, $args)*)),
            AbstractWeight::Factored(inner) => wrap_factored(<Arc<FactoredWeight> as BackendOps>::$method(inner $(, $args)*)),
            AbstractWeight::FactoredValidate(inner) => wrap_factored_validate(<Arc<FactoredValidateWeight> as BackendOps>::$method(inner $(, $args)*)),
        }
    }};
}

macro_rules! dispatch_mut {
    ($self:expr, $method:ident $(, $args:expr)*) => {{
        match $self {
            AbstractWeight::Rs(inner) => <RangeSet as BackendOps>::$method(inner $(, $args)*),
            AbstractWeight::Bdd(inner) => <Arc<BddWeight> as BackendOps>::$method(inner $(, $args)*),
            AbstractWeight::BddBiodivine(inner) => <Arc<BddWeightBiodivine> as BackendOps>::$method(inner $(, $args)*),
            AbstractWeight::Factored(inner) => <Arc<FactoredWeight> as BackendOps>::$method(inner $(, $args)*),
            AbstractWeight::FactoredValidate(inner) => <Arc<FactoredValidateWeight> as BackendOps>::$method(inner $(, $args)*),
        }
    }};
}

macro_rules! dispatch_binary_to_weight {
    ($lhs:expr, $rhs:expr, $method:ident $(, $args:expr)*) => {{
        match ($lhs, $rhs) {
            (AbstractWeight::Rs(a), AbstractWeight::Rs(b)) => wrap_rs(<RangeSet as BackendOps>::$method(a, b $(, $args)*)),
            (AbstractWeight::Bdd(a), AbstractWeight::Bdd(b)) => wrap_bdd(<Arc<BddWeight> as BackendOps>::$method(a, b $(, $args)*)),
            (AbstractWeight::BddBiodivine(a), AbstractWeight::BddBiodivine(b)) => {
                wrap_bdd_biodivine(<Arc<BddWeightBiodivine> as BackendOps>::$method(a, b $(, $args)*))
            }
            (AbstractWeight::Factored(a), AbstractWeight::Factored(b)) => {
                wrap_factored(<Arc<FactoredWeight> as BackendOps>::$method(a, b $(, $args)*))
            }
            (AbstractWeight::FactoredValidate(a), AbstractWeight::FactoredValidate(b)) => {
                wrap_factored_validate(<Arc<FactoredValidateWeight> as BackendOps>::$method(a, b $(, $args)*))
            }
            _ => panic!("Cross-backend operation is not supported"),
        }
    }};
}

macro_rules! dispatch_binary_value {
    ($lhs:expr, $rhs:expr, $method:ident $(, $args:expr)*) => {{
        match ($lhs, $rhs) {
            (AbstractWeight::Rs(a), AbstractWeight::Rs(b)) => <RangeSet as BackendOps>::$method(a, b $(, $args)*),
            (AbstractWeight::Bdd(a), AbstractWeight::Bdd(b)) => <Arc<BddWeight> as BackendOps>::$method(a, b $(, $args)*),
            (AbstractWeight::BddBiodivine(a), AbstractWeight::BddBiodivine(b)) => {
                <Arc<BddWeightBiodivine> as BackendOps>::$method(a, b $(, $args)*)
            }
            (AbstractWeight::Factored(a), AbstractWeight::Factored(b)) => {
                <Arc<FactoredWeight> as BackendOps>::$method(a, b $(, $args)*)
            }
            (AbstractWeight::FactoredValidate(a), AbstractWeight::FactoredValidate(b)) => {
                <Arc<FactoredValidateWeight> as BackendOps>::$method(a, b $(, $args)*)
            }
            _ => panic!("Cross-backend operation is not supported"),
        }
    }};
}

macro_rules! dispatch_backend_ctor {
    ($method:ident $(, $args:expr)*) => {{
        match get_weight_backend() {
            WeightBackend::RangeSet => wrap_rs(<RangeSet as BackendOps>::$method($($args),*)),
            WeightBackend::Bdd => wrap_bdd(<Arc<BddWeight> as BackendOps>::$method($($args),*)),
            WeightBackend::BddBiodivine => wrap_bdd_biodivine(<Arc<BddWeightBiodivine> as BackendOps>::$method($($args),*)),
            WeightBackend::Factored => wrap_factored(<Arc<FactoredWeight> as BackendOps>::$method($($args),*)),
            WeightBackend::FactoredValidate => wrap_factored_validate(<Arc<FactoredValidateWeight> as BackendOps>::$method($($args),*)),
        }
    }};
}

// ============================================================================
// BackendOps implementations
// ============================================================================

impl BackendOps for RangeSet {
    fn zeros() -> Self { RangeSet::zeros() }
    fn from_item(item: usize) -> Self { RangeSet::from_item(item) }
    fn from_ranges(ranges: &[(usize, usize)]) -> Self { RangeSet::from_ranges(ranges) }
    fn from_rsb(rsb: RangeSetBlaze<usize>) -> Self { RangeSet::from_rsb(rsb) }
    fn ones(len: usize) -> Self { RangeSet::ones(len) }
    fn from_token_set_all_tsids(tokens: RangeSetBlaze<usize>, dims: WeightDimensions) -> Self {
        let expanded = crate::dwa_i32::weight_expansion::expand_rsb(&tokens, dims.num_tsids);
        RangeSet::from_rsb(expanded)
    }
    fn from_token_set_specific_tsid(tokens: RangeSetBlaze<usize>, tsid: usize, dims: WeightDimensions) -> Self {
        let positions: RangeSetBlaze<usize> = tokens.iter().map(|t| t * dims.num_tsids + tsid).collect();
        RangeSet::from_rsb(positions)
    }
    fn tsid_columns_with_dims(tsids: Vec<usize>, num_tsids: usize, num_tokens: usize) -> Self {
        let mut rsb = RangeSetBlaze::new();
        for n in 0..num_tokens {
            let offset = n * num_tsids;
            for &tsid in &tsids {
                rsb.insert(tsid + offset);
            }
        }
        RangeSet::from_rsb(rsb)
    }

    fn num_ranges(&self) -> usize { RangeSetInner::num_ranges(self) }
    fn len(&self) -> usize { RangeSetInner::len(self) }
    fn is_empty(&self) -> bool { RangeSetInner::is_empty(self) }
    fn is_all_fast(&self) -> bool { RangeSetInner::is_all_fast(self) }
    fn contains(&self, pos: usize) -> bool { RangeSetInner::contains(self, pos) }
    fn is_disjoint(&self, other: &Self) -> bool { RangeSetInner::is_disjoint(self, other) }
    fn is_subset_of(&self, other: &Self) -> bool { RangeSet::is_subset_of(self, other) }
    fn min_item(&self) -> Option<usize> { RangeSetInner::min_item(self) }
    fn max_item(&self) -> Option<usize> { RangeSetInner::max_item(self) }
    fn fp(&self) -> u64 { RangeSet::fast_hash(self) }
    fn intern_id(&self) -> usize { RangeSet::intern_id(self) }

    fn iter<'a>(&'a self) -> Box<dyn Iterator<Item = usize> + 'a> { Box::new(self.rsb.iter()) }
    fn iter_up_to<'a>(&'a self, max: usize) -> Box<dyn Iterator<Item = usize> + 'a> {
        Box::new(RangeSetInner::iter_up_to(self, max))
    }
    fn ranges<'a>(&'a self) -> Box<dyn Iterator<Item = std::ops::RangeInclusive<usize>> + 'a> { Box::new(self.rsb.ranges()) }
    fn to_rangeset(&self) -> RangeSet { self.clone() }
    fn to_rsb(&self) -> RangeSetBlaze<usize> { self.rsb.clone() }

    fn union(&self, other: &Self) -> Self { self | other }
    fn intersection(&self, other: &Self) -> Self { self & other }
    fn subtract(&self, other: &Self) -> Self { self - other }
    fn clip_max(&self, max: usize) -> Self {
        let mut clipped = self.clone();
        RangeSet::clip_max(&mut clipped, max);
        clipped
    }
    fn insert(&mut self, item: usize) { RangeSet::insert(self, item); }
    fn remove(&mut self, item: usize) { RangeSet::remove(self, item); }

    fn fmt_display(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.is_empty() {
            write!(f, "∅")
        } else {
            let ranges: Vec<_> = self.rsb.ranges().collect();
            if ranges.len() <= 5 {
                let strs: Vec<String> = ranges.iter().map(|r| format!("{}..={}", r.start(), r.end())).collect();
                write!(f, "[{}]", strs.join(", "))
            } else {
                write!(f, "[{} ranges]", ranges.len())
            }
        }
    }
}

impl BackendOps for Arc<BddWeight> {
    fn zeros() -> Self {
        let dims = get_weight_dimensions();
        Arc::new(BddWeight::empty(dims.num_tsids as u16, dims.num_tokens as u16))
    }
    fn from_item(item: usize) -> Self {
        let dims = get_weight_dimensions();
        Arc::new(BddWeight::from_ranges(std::iter::once((item, item)), dims.num_tsids as u16, dims.num_tokens as u16))
    }
    fn from_ranges(ranges: &[(usize, usize)]) -> Self {
        let dims = get_weight_dimensions();
        Arc::new(BddWeight::from_ranges(ranges.iter().copied(), dims.num_tsids as u16, dims.num_tokens as u16))
    }
    fn from_rsb(rsb: RangeSetBlaze<usize>) -> Self {
        let dims = get_weight_dimensions();
        let ranges: Vec<(usize, usize)> = rsb.ranges().map(|r| (*r.start(), *r.end())).collect();
        Arc::new(BddWeight::from_ranges(ranges.into_iter(), dims.num_tsids as u16, dims.num_tokens as u16))
    }
    fn ones(len: usize) -> Self {
        let dims = get_weight_dimensions();
        Arc::new(BddWeight::from_ranges(std::iter::once((0, len.saturating_sub(1))), dims.num_tsids as u16, dims.num_tokens as u16))
    }
    fn from_token_set_all_tsids(tokens: RangeSetBlaze<usize>, dims: WeightDimensions) -> Self {
        let expanded = crate::dwa_i32::weight_expansion::expand_rsb(&tokens, dims.num_tsids);
        Self::from_rsb(expanded)
    }
    fn from_token_set_specific_tsid(tokens: RangeSetBlaze<usize>, tsid: usize, dims: WeightDimensions) -> Self {
        let positions: RangeSetBlaze<usize> = tokens.iter().map(|t| t * dims.num_tsids + tsid).collect();
        Self::from_rsb(positions)
    }
    fn tsid_columns_with_dims(tsids: Vec<usize>, num_tsids: usize, num_tokens: usize) -> Self {
        let mut rsb = RangeSetBlaze::new();
        for n in 0..num_tokens {
            let offset = n * num_tsids;
            for &tsid in &tsids {
                rsb.insert(tsid + offset);
            }
        }
        Self::from_rsb(rsb)
    }

    fn num_ranges(&self) -> usize { self.as_ref().to_rangeset().ranges_len() as usize }
    fn len(&self) -> usize { self.as_ref().len() }
    fn is_empty(&self) -> bool { self.as_ref().is_empty() }
    fn is_all_fast(&self) -> bool { self.as_ref().is_full() }
    fn contains(&self, pos: usize) -> bool { self.as_ref().contains_pos(pos) }
    fn is_disjoint(&self, other: &Self) -> bool { self.as_ref().intersection(other.as_ref()).is_empty() }
    fn is_subset_of(&self, other: &Self) -> bool { self.as_ref().subtract(other.as_ref()).is_empty() }
    fn min_item(&self) -> Option<usize> { self.as_ref().iter().next() }
    fn max_item(&self) -> Option<usize> { self.as_ref().iter().last() }
    fn fp(&self) -> u64 { Arc::as_ptr(self) as u64 }
    fn intern_id(&self) -> usize { Arc::as_ptr(self) as usize }

    fn iter<'a>(&'a self) -> Box<dyn Iterator<Item = usize> + 'a> { Box::new(self.as_ref().iter()) }
    fn iter_up_to<'a>(&'a self, max: usize) -> Box<dyn Iterator<Item = usize> + 'a> { Box::new(self.as_ref().iter().take_while(move |&p| p <= max)) }
    fn ranges<'a>(&'a self) -> Box<dyn Iterator<Item = std::ops::RangeInclusive<usize>> + 'a> {
        Box::new(self.as_ref().to_rangeset().into_ranges())
    }
    fn to_rangeset(&self) -> RangeSet { RangeSet::from_rsb(self.as_ref().to_rangeset()) }
    fn to_rsb(&self) -> RangeSetBlaze<usize> { self.as_ref().to_rangeset() }

    fn union(&self, other: &Self) -> Self { Arc::new(self.as_ref().union(other.as_ref())) }
    fn intersection(&self, other: &Self) -> Self { Arc::new(self.as_ref().intersection(other.as_ref())) }
    fn subtract(&self, other: &Self) -> Self { Arc::new(self.as_ref().subtract(other.as_ref())) }
    fn clip_max(&self, max: usize) -> Self {
        let ranges: Vec<_> = self.as_ref().to_rangeset().into_ranges()
            .filter_map(|r| {
                let start = *r.start();
                let end = (*r.end()).min(max);
                if start <= max { Some((start, end)) } else { None }
            })
            .collect();
        let dims = get_weight_dimensions();
        Arc::new(BddWeight::from_ranges(ranges.into_iter(), dims.num_tsids as u16, dims.num_tokens as u16))
    }
    fn insert(&mut self, _item: usize) { panic!("insert() not supported for BDD backend"); }
    fn remove(&mut self, _item: usize) { panic!("remove() not supported for BDD backend"); }

    fn fmt_display(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let ranges: Vec<_> = self.as_ref().to_rangeset().into_ranges().collect();
        if ranges.is_empty() {
            write!(f, "∅")
        } else if ranges.len() <= 5 {
            let strs: Vec<String> = ranges.iter().map(|r| format!("{}..={}", r.start(), r.end())).collect();
            write!(f, "[{}]", strs.join(", "))
        } else {
            write!(f, "[{} ranges]", ranges.len())
        }
    }
}

impl BackendOps for Arc<BddWeightBiodivine> {
    fn zeros() -> Self {
        let dims = get_weight_dimensions();
        Arc::new(BddWeightBiodivine::empty(dims.num_tsids as u16, dims.num_tokens as u16))
    }
    fn from_item(item: usize) -> Self {
        let dims = get_weight_dimensions();
        Arc::new(BddWeightBiodivine::from_ranges(std::iter::once((item, item)), dims.num_tsids as u16, dims.num_tokens as u16))
    }
    fn from_ranges(ranges: &[(usize, usize)]) -> Self {
        let dims = get_weight_dimensions();
        Arc::new(BddWeightBiodivine::from_ranges(ranges.iter().copied(), dims.num_tsids as u16, dims.num_tokens as u16))
    }
    fn from_rsb(rsb: RangeSetBlaze<usize>) -> Self {
        let dims = get_weight_dimensions();
        let ranges: Vec<(usize, usize)> = rsb.ranges().map(|r| (*r.start(), *r.end())).collect();
        Arc::new(BddWeightBiodivine::from_ranges(ranges.into_iter(), dims.num_tsids as u16, dims.num_tokens as u16))
    }
    fn ones(len: usize) -> Self {
        let dims = get_weight_dimensions();
        Arc::new(BddWeightBiodivine::from_ranges(std::iter::once((0, len.saturating_sub(1))), dims.num_tsids as u16, dims.num_tokens as u16))
    }
    fn from_token_set_all_tsids(tokens: RangeSetBlaze<usize>, dims: WeightDimensions) -> Self {
        let expanded = crate::dwa_i32::weight_expansion::expand_rsb(&tokens, dims.num_tsids);
        Self::from_rsb(expanded)
    }
    fn from_token_set_specific_tsid(tokens: RangeSetBlaze<usize>, tsid: usize, dims: WeightDimensions) -> Self {
        let positions: RangeSetBlaze<usize> = tokens.iter().map(|t| t * dims.num_tsids + tsid).collect();
        Self::from_rsb(positions)
    }
    fn tsid_columns_with_dims(tsids: Vec<usize>, num_tsids: usize, num_tokens: usize) -> Self {
        Arc::new(BddWeightBiodivine::tsid_columns(tsids.into_iter().map(|t| t as u16), num_tsids as u16, num_tokens as u16))
    }

    fn num_ranges(&self) -> usize { self.as_ref().to_rangeset().ranges_len() as usize }
    fn len(&self) -> usize { self.len() }
    fn is_empty(&self) -> bool { self.is_empty() }
    fn is_all_fast(&self) -> bool { self.is_full() }
    fn contains(&self, pos: usize) -> bool { self.contains_pos(pos) }
    fn is_disjoint(&self, other: &Self) -> bool { self.intersection(other).is_empty() }
    fn is_subset_of(&self, other: &Self) -> bool { self.subtract(other).is_empty() }
    fn min_item(&self) -> Option<usize> { self.iter().next() }
    fn max_item(&self) -> Option<usize> { self.iter().last() }
    fn fp(&self) -> u64 { Arc::as_ptr(self) as u64 }
    fn intern_id(&self) -> usize { Arc::as_ptr(self) as usize }

    fn iter<'a>(&'a self) -> Box<dyn Iterator<Item = usize> + 'a> { Box::new(self.iter()) }
    fn iter_up_to<'a>(&'a self, max: usize) -> Box<dyn Iterator<Item = usize> + 'a> { Box::new(self.iter().take_while(move |&p| p <= max)) }
    fn ranges<'a>(&'a self) -> Box<dyn Iterator<Item = std::ops::RangeInclusive<usize>> + 'a> {
        Box::new(self.as_ref().to_rangeset().into_ranges())
    }
    fn to_rangeset(&self) -> RangeSet { RangeSet::from_rsb(self.as_ref().to_rangeset()) }
    fn to_rsb(&self) -> RangeSetBlaze<usize> { self.as_ref().to_rangeset() }

    fn union(&self, other: &Self) -> Self { Arc::new(self.as_ref().union(other.as_ref())) }
    fn intersection(&self, other: &Self) -> Self { Arc::new(self.as_ref().intersection(other.as_ref())) }
    fn subtract(&self, other: &Self) -> Self { Arc::new(self.as_ref().subtract(other.as_ref())) }
    fn clip_max(&self, max: usize) -> Self {
        let ranges: Vec<_> = self.as_ref().to_rangeset().into_ranges()
            .filter_map(|r| {
                let start = *r.start();
                let end = (*r.end()).min(max);
                if start <= max { Some((start, end)) } else { None }
            })
            .collect();
        let dims = get_weight_dimensions();
        Arc::new(BddWeightBiodivine::from_ranges(ranges.into_iter(), dims.num_tsids as u16, dims.num_tokens as u16))
    }
    fn insert(&mut self, _item: usize) { panic!("insert() not supported for BDD backend"); }
    fn remove(&mut self, _item: usize) { panic!("remove() not supported for BDD backend"); }

    fn fmt_display(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let ranges: Vec<_> = self.as_ref().to_rangeset().into_ranges().collect();
        if ranges.is_empty() {
            write!(f, "∅")
        } else if ranges.len() <= 5 {
            let strs: Vec<String> = ranges.iter().map(|r| format!("{}..={}", r.start(), r.end())).collect();
            write!(f, "[{}]", strs.join(", "))
        } else {
            write!(f, "[{} ranges]", ranges.len())
        }
    }
}

impl BackendOps for Arc<FactoredWeight> {
    fn zeros() -> Self {
        let dims = get_weight_dimensions();
        Arc::new(FactoredWeight::empty(dims.num_tsids as u16))
    }
    fn from_item(item: usize) -> Self {
        let dims = get_weight_dimensions();
        Arc::new(FactoredWeight::from_1d_ranges(std::iter::once((item, item)), dims.num_tsids))
    }
    fn from_ranges(ranges: &[(usize, usize)]) -> Self {
        let dims = get_weight_dimensions();
        Arc::new(FactoredWeight::from_1d_ranges(ranges.iter().copied(), dims.num_tsids))
    }
    fn from_rsb(rsb: RangeSetBlaze<usize>) -> Self {
        let dims = get_weight_dimensions();
        Arc::new(FactoredWeight::from_rsb(&rsb, dims.num_tsids))
    }
    fn ones(len: usize) -> Self {
        let dims = get_weight_dimensions();
        Arc::new(FactoredWeight::from_1d_ranges(std::iter::once((0, len.saturating_sub(1))), dims.num_tsids))
    }
    fn from_token_set_all_tsids(tokens: RangeSetBlaze<usize>, dims: WeightDimensions) -> Self {
        Arc::new(FactoredWeight::from_token_set_all_tsids(tokens, dims.num_tsids as u16))
    }
    fn from_token_set_specific_tsid(tokens: RangeSetBlaze<usize>, tsid: usize, dims: WeightDimensions) -> Self {
        Arc::new(FactoredWeight::from_token_set_specific_tsid(tokens, tsid, dims.num_tsids as u16))
    }
    fn tsid_columns_with_dims(tsids: Vec<usize>, num_tsids: usize, num_tokens: usize) -> Self {
        let tsid_set: RangeSetBlaze<u16> = tsids.into_iter().map(|t| t as u16).collect();
        let token_set: RangeSetBlaze<u16> = (0..num_tokens as u16).collect();
        Arc::new(FactoredWeight::from_product(token_set, tsid_set, num_tsids as u16))
    }

    fn num_ranges(&self) -> usize { self.as_ref().total_ranges() }
    fn len(&self) -> usize { self.as_ref().len() }
    fn is_empty(&self) -> bool { self.as_ref().is_empty() }
    fn is_all_fast(&self) -> bool { self.as_ref().is_full() }
    fn contains(&self, pos: usize) -> bool { self.as_ref().contains_pos(pos) }
    fn is_disjoint(&self, other: &Self) -> bool { self.as_ref().intersection(other.as_ref()).is_empty() }
    fn is_subset_of(&self, other: &Self) -> bool { self.as_ref().is_subset_of(other.as_ref()) }
    fn min_item(&self) -> Option<usize> { self.as_ref().min_position() }
    fn max_item(&self) -> Option<usize> { self.as_ref().max_position() }
    fn fp(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        self.as_ref().hash_2d(&mut hasher);
        hasher.finish()
    }
    fn intern_id(&self) -> usize { Arc::as_ptr(self) as usize }

    fn iter<'a>(&'a self) -> Box<dyn Iterator<Item = usize> + 'a> { Box::new(self.as_ref().iter_positions()) }
    fn iter_up_to<'a>(&'a self, max: usize) -> Box<dyn Iterator<Item = usize> + 'a> { Box::new(self.as_ref().iter_positions_up_to(max)) }
    fn ranges<'a>(&'a self) -> Box<dyn Iterator<Item = std::ops::RangeInclusive<usize>> + 'a> {
        Box::new(self.as_ref().expand_impl().into_ranges())
    }
    fn to_rangeset(&self) -> RangeSet { RangeSet::from_rsb(self.as_ref().expand_impl()) }
    fn to_rsb(&self) -> RangeSetBlaze<usize> { self.as_ref().expand_impl() }

    fn union(&self, other: &Self) -> Self { Arc::new(self.as_ref().union(other.as_ref())) }
    fn intersection(&self, other: &Self) -> Self { Arc::new(self.as_ref().intersection(other.as_ref())) }
    fn subtract(&self, other: &Self) -> Self { Arc::new(self.as_ref().subtract(other.as_ref())) }
    fn clip_max(&self, max: usize) -> Self { Arc::new(self.as_ref().clip_max(max)) }
    fn insert(&mut self, item: usize) { *self = Arc::new(self.as_ref().insert_pos(item)); }
    fn remove(&mut self, item: usize) { *self = Arc::new(self.as_ref().remove_pos(item)); }

    fn fmt_display(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let num_terms = self.num_terms();
        if self.is_empty() {
            write!(f, "∅")
        } else {
            write!(f, "[Factored: {} terms]", num_terms)
        }
    }
}

impl BackendOps for Arc<FactoredValidateWeight> {
    fn zeros() -> Self {
        let dims = get_weight_dimensions();
        let factored = FactoredWeight::empty(dims.num_tsids as u16);
        let rangeset = RangeSet::zeros();
        Arc::new(FactoredValidateWeight::new(factored, rangeset))
    }
    fn from_item(item: usize) -> Self {
        let dims = get_weight_dimensions();
        let factored = FactoredWeight::from_1d_ranges(std::iter::once((item, item)), dims.num_tsids);
        let rangeset = RangeSet::from_item(item);
        Arc::new(FactoredValidateWeight::new(factored, rangeset))
    }
    fn from_ranges(ranges: &[(usize, usize)]) -> Self {
        let dims = get_weight_dimensions();
        let factored = FactoredWeight::from_1d_ranges(ranges.iter().copied(), dims.num_tsids);
        let rangeset = RangeSet::from_ranges(ranges);
        Arc::new(FactoredValidateWeight::new(factored, rangeset))
    }
    fn from_rsb(rsb: RangeSetBlaze<usize>) -> Self {
        let dims = get_weight_dimensions();
        let factored = FactoredWeight::from_rsb(&rsb, dims.num_tsids);
        let rangeset = RangeSet::from_rsb(rsb);
        Arc::new(FactoredValidateWeight::new(factored, rangeset))
    }
    fn ones(len: usize) -> Self {
        let dims = get_weight_dimensions();
        let factored = FactoredWeight::from_1d_ranges(std::iter::once((0, len.saturating_sub(1))), dims.num_tsids);
        let rangeset = RangeSet::ones(len);
        Arc::new(FactoredValidateWeight::new(factored, rangeset))
    }
    fn from_token_set_all_tsids(tokens: RangeSetBlaze<usize>, dims: WeightDimensions) -> Self {
        let factored = FactoredWeight::from_token_set_all_tsids(tokens.clone(), dims.num_tsids as u16);
        let expanded = crate::dwa_i32::weight_expansion::expand_rsb(&tokens, dims.num_tsids);
        let rangeset = RangeSet::from_rsb(expanded);
        Arc::new(FactoredValidateWeight::new(factored, rangeset))
    }
    fn from_token_set_specific_tsid(tokens: RangeSetBlaze<usize>, tsid: usize, dims: WeightDimensions) -> Self {
        let factored = FactoredWeight::from_token_set_specific_tsid(tokens.clone(), tsid, dims.num_tsids as u16);
        let positions: RangeSetBlaze<usize> = tokens.iter().map(|t| t * dims.num_tsids + tsid).collect();
        let rangeset = RangeSet::from_rsb(positions);
        Arc::new(FactoredValidateWeight::new(factored, rangeset))
    }
    fn tsid_columns_with_dims(tsids: Vec<usize>, num_tsids: usize, num_tokens: usize) -> Self {
        let tsid_set: RangeSetBlaze<u16> = tsids.iter().map(|t| *t as u16).collect();
        let token_set: RangeSetBlaze<u16> = (0..num_tokens as u16).collect();
        let factored = FactoredWeight::from_product(token_set, tsid_set, num_tsids as u16);
        let mut rsb = RangeSetBlaze::new();
        for n in 0..num_tokens {
            let offset = n * num_tsids;
            for &tsid in &tsids {
                rsb.insert(tsid + offset);
            }
        }
        let rangeset = RangeSet::from_rsb(rsb);
        Arc::new(FactoredValidateWeight::new(factored, rangeset))
    }

    fn num_ranges(&self) -> usize { self.rangeset().num_ranges() }
    fn len(&self) -> usize { self.rangeset().len() }
    fn is_empty(&self) -> bool { self.rangeset().is_empty() }
    fn is_all_fast(&self) -> bool { self.rangeset().is_all_fast() }
    fn contains(&self, pos: usize) -> bool { self.rangeset().contains(pos) }
    fn is_disjoint(&self, other: &Self) -> bool { self.rangeset().is_disjoint(other.rangeset()) }
    fn is_subset_of(&self, other: &Self) -> bool { self.rangeset().is_subset_of(other.rangeset()) }
    fn min_item(&self) -> Option<usize> { self.rangeset().min_item() }
    fn max_item(&self) -> Option<usize> { self.rangeset().max_item() }
    fn fp(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        self.factored().hash_2d(&mut hasher);
        hasher.finish()
    }
    fn intern_id(&self) -> usize { Arc::as_ptr(self) as usize }

    fn iter<'a>(&'a self) -> Box<dyn Iterator<Item = usize> + 'a> { Box::new(self.rangeset().rsb.iter()) }
    fn iter_up_to<'a>(&'a self, max: usize) -> Box<dyn Iterator<Item = usize> + 'a> { Box::new(self.rangeset().iter_up_to(max)) }
    fn ranges<'a>(&'a self) -> Box<dyn Iterator<Item = std::ops::RangeInclusive<usize>> + 'a> { Box::new(self.rangeset().rsb.ranges()) }
    fn to_rangeset(&self) -> RangeSet { self.rangeset().clone() }
    fn to_rsb(&self) -> RangeSetBlaze<usize> { self.rangeset().rsb.clone() }

    fn union(&self, other: &Self) -> Self {
        let factored = self.factored().union(other.factored());
        let rangeset = self.rangeset() | other.rangeset();
        Arc::new(FactoredValidateWeight::new(factored, rangeset))
    }
    fn intersection(&self, other: &Self) -> Self {
        let factored = self.factored().intersection(other.factored());
        let rangeset = self.rangeset() & other.rangeset();
        Arc::new(FactoredValidateWeight::new(factored, rangeset))
    }
    fn subtract(&self, other: &Self) -> Self {
        let factored = self.factored().subtract(other.factored());
        let rangeset = self.rangeset() - other.rangeset();
        Arc::new(FactoredValidateWeight::new(factored, rangeset))
    }
    fn clip_max(&self, max: usize) -> Self {
        let factored = self.factored().clip_max(max);
        let mut rangeset = self.rangeset().clone();
        rangeset.clip_max(max);
        Arc::new(FactoredValidateWeight::new(factored, rangeset))
    }
    fn insert(&mut self, item: usize) {
        let factored = self.factored().insert_pos(item);
        let mut rangeset = self.rangeset().clone();
        rangeset.insert(item);
        *self = Arc::new(FactoredValidateWeight::new(factored, rangeset));
    }
    fn remove(&mut self, item: usize) {
        let factored = self.factored().remove_pos(item);
        let mut rangeset = self.rangeset().clone();
        rangeset.remove(item);
        *self = Arc::new(FactoredValidateWeight::new(factored, rangeset));
    }

    fn fmt_display(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.rangeset().is_empty() {
            write!(f, "∅")
        } else {
            write!(f, "[FactoredValidate: {} terms, {} ranges]", self.factored().num_terms(), self.rangeset().num_ranges())
        }
    }
}

impl AbstractWeight {
    // ------------------------------------------------------------------------
    // Constructors
    // ------------------------------------------------------------------------

    /// Create an empty weight (accepts nothing).
    pub fn zeros() -> Self {
        dispatch_backend_ctor!(zeros)
    }

    /// Create a weight containing a single position.
    pub fn from_item(item: usize) -> Self {
        dispatch_backend_ctor!(from_item, item)
    }

    /// Create a weight from ranges.
    pub fn from_ranges(ranges: &[(usize, usize)]) -> Self {
        dispatch_backend_ctor!(from_ranges, ranges)
    }

    /// Create a weight from a RangeSetBlaze.
    pub fn from_rsb(rsb: RangeSetBlaze<usize>) -> Self {
        dispatch_backend_ctor!(from_rsb, rsb)
    }
    
    /// Create a weight from a token set (N-space) × all TSIDs.
    /// This is efficient for weight-heavy mode where we have tokens in N-space
    /// and want the weight to cover all TSID values.
    /// 
    /// For FactoredWeight, this is O(n_ranges) instead of O(n_tokens × n_tsids).
    pub fn from_token_set_all_tsids(tokens: RangeSetBlaze<usize>) -> Self {
        let dims = get_weight_dimensions();
        dispatch_backend_ctor!(from_token_set_all_tsids, tokens, dims)
    }
    
    /// Create a weight from a token set (N-space) × specific TSID.
    /// This is efficient for weight-heavy precomputation where we know the exact tsid.
    /// 
    /// For FactoredWeight, this is O(n_ranges) instead of O(n_tokens × n_tsids).
    pub fn from_token_set_specific_tsid(tokens: RangeSetBlaze<usize>, tsid: usize) -> Self {
        let dims = get_weight_dimensions();
        dispatch_backend_ctor!(from_token_set_specific_tsid, tokens, tsid, dims)
    }

    /// Create a weight for positions 0..=len-1.
    pub fn ones(len: usize) -> Self {
        dispatch_backend_ctor!(ones, len)
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
        let tsid_vec: Vec<usize> = tsids.into_iter().collect();
        dispatch_backend_ctor!(tsid_columns_with_dims, tsid_vec, num_tsids, num_tokens)
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
        dispatch_ref!(self, is_empty)
    }

    /// Fast check if this weight is "all" (accepts everything).
    pub fn is_all_fast(&self) -> bool {
        dispatch_ref!(self, is_all_fast)
    }

    /// Check if a position is contained in this weight.
    pub fn contains(&self, pos: usize) -> bool {
        dispatch_ref!(self, contains, pos)
    }

    /// Check if two weights are disjoint.
    pub fn is_disjoint(&self, other: &Self) -> bool {
        dispatch_binary_value!(self, other, is_disjoint)
    }

    /// Get the number of ranges (only meaningful for RangeSet backend).
    /// For Factored backend, returns total_ranges() which is an approximation.
    pub fn num_ranges(&self) -> usize {
        dispatch_ref!(self, num_ranges)
    }

    /// Get the number of elements in this weight.
    pub fn len(&self) -> usize {
        dispatch_ref!(self, len)
    }

    /// Get a fast hash/fingerprint for this weight.
    /// Used for quick equality checking in interning.
    pub fn fp(&self) -> u64 {
        dispatch_ref!(self, fp)
    }

    /// Get the minimum item in this weight.
    pub fn min_item(&self) -> Option<usize> {
        dispatch_ref!(self, min_item)
    }

    /// Get the maximum item in this weight.
    pub fn max_item(&self) -> Option<usize> {
        dispatch_ref!(self, max_item)
    }

    /// Insert an item into this weight.
    pub fn insert(&mut self, item: usize) {
        dispatch_mut!(self, insert, item)
    }

    /// Remove an item from this weight.
    pub fn remove(&mut self, item: usize) {
        dispatch_mut!(self, remove, item)
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

        dispatch_binary_value!(self, other, is_subset_of)
    }

    /// Clip all values above a maximum, returning a new weight.
    pub fn clip_max(&self, max: usize) -> Self {
        dispatch_ref_to_weight!(self, clip_max, max)
    }

    // ------------------------------------------------------------------------
    // Conversions
    // ------------------------------------------------------------------------

    /// Convert to RangeSet (for compatibility with existing code).
    /// NOTE: For Factored backend, this requires 1D expansion.
    pub fn to_rangeset(&self) -> RangeSet {
        dispatch_ref!(self, to_rangeset)
    }

    /// Get the underlying RangeSetBlaze.
    /// NOTE: For Factored backend, this requires 1D expansion.
    pub fn to_rsb(&self) -> RangeSetBlaze<usize> {
        dispatch_ref!(self, to_rsb)
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
        dispatch_ref!(self, iter)
    }

    /// Iterate over accepted positions up to a maximum value.
    pub fn iter_up_to(&self, max: usize) -> Box<dyn Iterator<Item = usize> + '_> {
        dispatch_ref!(self, iter_up_to, max)
    }

    /// Iterate over ranges.
    /// NOTE: For Factored backend, this requires 1D expansion.
    pub fn ranges(&self) -> Box<dyn Iterator<Item = std::ops::RangeInclusive<usize>> + '_> {
        dispatch_ref!(self, ranges)
    }

    // ------------------------------------------------------------------------
    // Access to internal RangeSetBlaze (for compatibility)
    // ------------------------------------------------------------------------

    /// Access the internal RangeSetBlaze (panics for BDD backends).
    /// This is provided for compatibility with existing code that accesses `.rsb`.
    pub fn rsb(&self) -> &RangeSetBlaze<usize> {
        match self {
            Self::Rs(rs) => &rs.rsb,
            Self::FactoredValidate(fw) => &fw.rangeset().rsb,
            _ => {
                panic!("Cannot access rsb on non-RangeSet weight - use to_rsb() instead")
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
        dispatch_ref!(self, intern_id)
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

        dispatch_binary_to_weight!(self, rhs, union)
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

        dispatch_binary_to_weight!(self, rhs, intersection)
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

impl Sub for &AbstractWeight {
    type Output = AbstractWeight;

    fn sub(self, rhs: Self) -> Self::Output {
        dispatch_binary_to_weight!(self, rhs, subtract)
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
            (Self::BddBiodivine(a), Self::BddBiodivine(b)) => a == b,
            (Self::Factored(a), Self::Factored(b)) => a == b,
            (Self::FactoredValidate(a), Self::FactoredValidate(b)) => a.rangeset() == b.rangeset(),
            _ => panic!("Cross-backend equality is not supported"),
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
        if std::mem::discriminant(self) != std::mem::discriminant(other) {
            panic!("Cross-backend ordering is not supported");
        }
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
            Self::FactoredValidate(fw) => write!(
                f,
                "Weight::FactoredValidate({} terms, {} ranges)",
                fw.factored().num_terms(),
                fw.rangeset().num_ranges()
            ),
        }
    }
}

impl Display for AbstractWeight {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        dispatch_ref!(self, fmt_display, f)
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

    fn ensure_test_dims() {
        // BDD/factored backends require dimensions for correct position encoding.
        // Use a small domain that still covers the positions used in these tests.
        let dims = WeightDimensions::new(10, 5); // 10 tokens, 5 tsids -> domain size 50
        set_weight_dimensions(dims);
    }

    #[test]
    fn test_zeros_and_all() {
        ensure_test_dims();
        let z = AbstractWeight::zeros();
        assert!(z.is_empty());
        assert!(!z.is_all_fast());

        let a = crate::dwa_i32::weight_all();
        assert!(!a.is_empty());
        assert!(a.is_all_fast());
    }

    #[test]
    fn test_from_item() {
        ensure_test_dims();
        let w = AbstractWeight::from_item(42);
        assert!(w.contains(42));
        assert!(!w.contains(41));
        assert!(!w.contains(43));
    }

    #[test]
    fn test_from_ranges() {
        ensure_test_dims();
        let w = AbstractWeight::from_ranges(&[(0, 10), (20, 30)]);
        assert!(w.contains(5));
        assert!(w.contains(25));
        assert!(!w.contains(15));
    }

    #[test]
    fn test_union() {
        ensure_test_dims();
        let a = AbstractWeight::from_ranges(&[(0, 5)]);
        let b = AbstractWeight::from_ranges(&[(10, 15)]);
        let c = &a | &b;

        assert!(c.contains(3));
        assert!(c.contains(12));
        assert!(!c.contains(7));
    }

    #[test]
    fn test_intersection() {
        ensure_test_dims();
        let a = AbstractWeight::from_ranges(&[(0, 10)]);
        let b = AbstractWeight::from_ranges(&[(5, 15)]);
        let c = &a & &b;

        assert!(c.contains(7));
        assert!(!c.contains(3));
        assert!(!c.contains(12));
    }

    #[test]
    fn test_subtraction() {
        ensure_test_dims();
        let a = AbstractWeight::from_ranges(&[(0, 10)]);
        let b = AbstractWeight::from_ranges(&[(5, 15)]);
        let c = &a - &b;

        assert!(c.contains(3));
        assert!(!c.contains(7));
    }

    #[test]
    fn test_complement() {
        ensure_test_dims();
        let a = AbstractWeight::from_ranges(&[(5, 10)]);
        let all = crate::dwa_i32::weight_all();
        let b = &all - &a;

        assert!(!b.contains(7));
        assert!(b.contains(3));
        assert!(b.contains(15));
    }

    #[test]
    fn test_equality() {
        ensure_test_dims();
        let a = AbstractWeight::from_ranges(&[(0, 10)]);
        let b = AbstractWeight::from_ranges(&[(0, 10)]);
        let c = AbstractWeight::from_ranges(&[(0, 11)]);

        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_from_iter() {
        ensure_test_dims();
        let w: AbstractWeight = vec![1, 3, 5, 7].into_iter().collect();
        assert!(w.contains(1));
        assert!(w.contains(5));
        assert!(!w.contains(2));
    }

    #[test]
    fn test_to_rangeset_roundtrip() {
        ensure_test_dims();
        let original = AbstractWeight::from_ranges(&[(5, 15), (25, 35)]);
        let rs = original.to_rangeset();
        let back = AbstractWeight::from(rs);

        assert_eq!(original, back);
    }

    #[test]
    fn test_factored_complement_via_rangeset() {
        ensure_test_dims();
        
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
        let all = crate::dwa_i32::weight_all();
        let w_not = &all - &w;
        
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
