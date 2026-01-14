//! Weight Storage Abstraction Framework.
//!
//! This module provides a trait-based framework for comparing different weight
//! storage implementations. It allows easy swapping between implementations
//! and benchmarking their performance characteristics.
//!
//! ## Implementations
//!
//! - `RangeSetStorage`: Current production storage using `RangeSet` (interned `RangeSetBlaze`)
//! - `BddStorage`: Per-weight BDD with TSID-first variable ordering
//! - `SharedBddStorage`: Shared BDD with node deduplication across weights
//!
//! ## Usage
//!
//! ```ignore
//! use crate::dwa_i32::weight_storage::*;
//!
//! // Create a storage factory
//! let factory = BddStorageFactory::new(4476, 4096);
//!
//! // Convert a weight to storage
//! let weight = Weight::from_iter([100..=200, 300..=400]);
//! let stored = factory.from_weight(&weight);
//!
//! // Query the storage
//! assert!(stored.contains_pos(150));
//!
//! // Get metrics
//! println!("Storage bytes: {}", stored.storage_bytes());
//! ```

use range_set_blaze::RangeSetBlaze;
use std::fmt::Debug;

use super::rangeset::RangeSet;
use super::bdd_weight::BddWeight;
use super::heavy_weight::WeightDimensions;

// ============================================================================
// Core Trait
// ============================================================================

/// A weight storage implementation that can store and query sets of positions.
pub trait WeightStorage: Clone + Debug + Send + Sync {
    /// Check if the storage is empty.
    fn is_empty(&self) -> bool;
    
    /// Check if the storage is full (accepts all valid positions).
    fn is_full(&self) -> bool;
    
    /// Check if a position is in the storage.
    fn contains_pos(&self, pos: usize) -> bool;
    
    /// Check if a (token, tsid) pair is in the storage.
    fn contains(&self, token: u16, tsid: u16) -> bool;
    
    /// Get the number of nodes/elements in storage-specific units.
    fn num_elements(&self) -> usize;
    
    /// Get the storage size in bytes.
    fn storage_bytes(&self) -> usize;
    
    /// Convert to a RangeSetBlaze (for compatibility).
    fn to_rangeset_blaze(&self) -> RangeSetBlaze<usize>;
    
    /// Get a name for this storage type.
    fn storage_name(&self) -> &'static str;
}

/// Factory for creating weight storage instances.
pub trait WeightStorageFactory: Clone {
    /// The storage type produced by this factory.
    type Storage: WeightStorage;
    
    /// Create storage from 1D ranges.
    fn from_ranges(&self, ranges: impl Iterator<Item = (usize, usize)>) -> Self::Storage;
    
    /// Create storage from a RangeSet.
    fn from_rangeset(&self, rs: &RangeSet) -> Self::Storage {
        self.from_ranges(rs.rsb.ranges().map(|r| (*r.start(), *r.end())))
    }
    
    /// Create empty storage.
    fn empty(&self) -> Self::Storage;
    
    /// Create full storage.
    fn full(&self) -> Self::Storage;
    
    /// Get the dimensions.
    fn dimensions(&self) -> WeightDimensions;
    
    /// Get the factory name.
    fn factory_name(&self) -> &'static str;
}

// ============================================================================
// RangeSet Storage (Current Production)
// ============================================================================

/// Storage using the current RangeSet implementation.
#[derive(Clone, Debug)]
pub struct RangeSetStorage {
    inner: RangeSet,
    dims: WeightDimensions,
}

impl WeightStorage for RangeSetStorage {
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
    
    fn is_full(&self) -> bool {
        self.inner.is_all_fast()
    }
    
    fn contains_pos(&self, pos: usize) -> bool {
        self.inner.contains(pos)
    }
    
    fn contains(&self, token: u16, tsid: u16) -> bool {
        let pos = token as usize * self.dims.num_tsids + tsid as usize;
        self.inner.contains(pos)
    }
    
    fn num_elements(&self) -> usize {
        self.inner.num_ranges()
    }
    
    fn storage_bytes(&self) -> usize {
        // Estimate: 2 usizes per range
        self.inner.num_ranges() * 16
    }
    
    fn to_rangeset_blaze(&self) -> RangeSetBlaze<usize> {
        self.inner.rsb.clone()
    }
    
    fn storage_name(&self) -> &'static str {
        "RangeSet"
    }
}

/// Factory for RangeSetStorage.
#[derive(Clone)]
pub struct RangeSetStorageFactory {
    dims: WeightDimensions,
}

impl RangeSetStorageFactory {
    /// Create a new factory with given dimensions.
    pub fn new(num_tsids: usize, num_tokens: usize) -> Self {
        Self {
            dims: WeightDimensions::new(num_tokens, num_tsids),
        }
    }
}

impl WeightStorageFactory for RangeSetStorageFactory {
    type Storage = RangeSetStorage;
    
    fn from_ranges(&self, ranges: impl Iterator<Item = (usize, usize)>) -> Self::Storage {
        let rsb: RangeSetBlaze<usize> = ranges.map(|(s, e)| s..=e).collect();
        RangeSetStorage {
            inner: RangeSet::from_rsb(rsb),
            dims: self.dims,
        }
    }
    
    fn empty(&self) -> Self::Storage {
        RangeSetStorage {
            inner: RangeSet::zeros(),
            dims: self.dims,
        }
    }
    
    fn full(&self) -> Self::Storage {
        RangeSetStorage {
            inner: RangeSet::all(),
            dims: self.dims,
        }
    }
    
    fn dimensions(&self) -> WeightDimensions {
        self.dims
    }
    
    fn factory_name(&self) -> &'static str {
        "RangeSetFactory"
    }
}

// ============================================================================
// Per-Weight BDD Storage
// ============================================================================

/// Storage using per-weight BDD with TSID-first ordering.
#[derive(Clone, Debug)]
pub struct BddStorage {
    inner: BddWeight,
}

impl WeightStorage for BddStorage {
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
    
    fn is_full(&self) -> bool {
        self.inner.is_full()
    }
    
    fn contains_pos(&self, pos: usize) -> bool {
        self.inner.contains_pos(pos)
    }
    
    fn contains(&self, token: u16, tsid: u16) -> bool {
        self.inner.contains(token, tsid)
    }
    
    fn num_elements(&self) -> usize {
        self.inner.num_nodes()
    }
    
    fn storage_bytes(&self) -> usize {
        self.inner.storage_bytes()
    }
    
    fn to_rangeset_blaze(&self) -> RangeSetBlaze<usize> {
        self.inner.to_rangeset()
    }
    
    fn storage_name(&self) -> &'static str {
        "PerWeightBDD"
    }
}

/// Factory for per-weight BDD storage.
#[derive(Clone)]
pub struct BddStorageFactory {
    tsid_dim: u16,
    token_dim: u16,
}

impl BddStorageFactory {
    /// Create a new factory with given dimensions.
    pub fn new(num_tsids: usize, num_tokens: usize) -> Self {
        Self {
            tsid_dim: num_tsids as u16,
            token_dim: num_tokens as u16,
        }
    }
}

impl WeightStorageFactory for BddStorageFactory {
    type Storage = BddStorage;
    
    fn from_ranges(&self, ranges: impl Iterator<Item = (usize, usize)>) -> Self::Storage {
        BddStorage {
            inner: BddWeight::from_ranges(ranges, self.tsid_dim, self.token_dim),
        }
    }
    
    fn empty(&self) -> Self::Storage {
        BddStorage {
            inner: BddWeight::empty(self.tsid_dim, self.token_dim),
        }
    }
    
    fn full(&self) -> Self::Storage {
        BddStorage {
            inner: BddWeight::full(self.tsid_dim, self.token_dim),
        }
    }
    
    fn dimensions(&self) -> WeightDimensions {
        WeightDimensions::new(self.token_dim as usize, self.tsid_dim as usize)
    }
    
    fn factory_name(&self) -> &'static str {
        "PerWeightBDDFactory"
    }
}

// ============================================================================
// Comparison Utilities
// ============================================================================

/// Compare two storage implementations for equality.
pub fn storages_equal<S1: WeightStorage, S2: WeightStorage>(s1: &S1, s2: &S2) -> bool {
    let rs1 = s1.to_rangeset_blaze();
    let rs2 = s2.to_rangeset_blaze();
    rs1 == rs2
}

/// Metrics for a storage implementation.
#[derive(Clone, Debug, Default)]
pub struct StorageMetrics {
    /// Name of the storage type.
    pub name: String,
    /// Number of weights analyzed.
    pub num_weights: usize,
    /// Total number of elements (nodes/ranges).
    pub total_elements: usize,
    /// Total storage bytes.
    pub total_bytes: usize,
    /// Maximum elements in any single weight.
    pub max_elements: usize,
    /// Average elements per weight.
    pub avg_elements: f64,
    /// Construction time in microseconds (if measured).
    pub construction_time_us: u64,
}

impl StorageMetrics {
    /// Create new metrics with given name.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            ..Default::default()
        }
    }
    
    /// Add a storage instance to the metrics.
    pub fn add<S: WeightStorage>(&mut self, storage: &S) {
        self.num_weights += 1;
        let elements = storage.num_elements();
        self.total_elements += elements;
        self.total_bytes += storage.storage_bytes();
        self.max_elements = self.max_elements.max(elements);
        self.avg_elements = self.total_elements as f64 / self.num_weights as f64;
    }
    
    /// Print a summary.
    pub fn print_summary(&self) {
        println!("=== {} Storage Metrics ===", self.name);
        println!("  Weights:       {}", self.num_weights);
        println!("  Total elements:{}", self.total_elements);
        println!("  Total bytes:   {} ({:.2} KB)", self.total_bytes, self.total_bytes as f64 / 1024.0);
        println!("  Max elements:  {}", self.max_elements);
        println!("  Avg elements:  {:.1}", self.avg_elements);
        if self.construction_time_us > 0 {
            println!("  Construct time:{} µs", self.construction_time_us);
        }
    }
}

/// Compare multiple storage factories on the same set of weights.
pub fn compare_storages<F1, F2>(
    factory1: &F1,
    factory2: &F2,
    weights: impl Iterator<Item = RangeSet>,
) -> (StorageMetrics, StorageMetrics, bool)
where
    F1: WeightStorageFactory,
    F2: WeightStorageFactory,
{
    let mut metrics1 = StorageMetrics::new(factory1.factory_name());
    let mut metrics2 = StorageMetrics::new(factory2.factory_name());
    let mut all_equal = true;
    
    for weight in weights {
        let s1 = factory1.from_rangeset(&weight);
        let s2 = factory2.from_rangeset(&weight);
        
        metrics1.add(&s1);
        metrics2.add(&s2);
        
        if !storages_equal(&s1, &s2) {
            all_equal = false;
        }
    }
    
    (metrics1, metrics2, all_equal)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_rangeset_factory() {
        let factory = RangeSetStorageFactory::new(100, 100);
        
        let empty = factory.empty();
        assert!(empty.is_empty());
        
        let full = factory.full();
        assert!(full.is_full());
        
        let single = factory.from_ranges(vec![(50, 100)].into_iter());
        assert!(single.contains_pos(75));
        assert!(!single.contains_pos(25));
    }
    
    #[test]
    fn test_bdd_factory() {
        let factory = BddStorageFactory::new(100, 100);
        
        let empty = factory.empty();
        assert!(empty.is_empty());
        
        let full = factory.full();
        assert!(full.is_full());
        
        let single = factory.from_ranges(vec![(50, 100)].into_iter());
        assert!(single.contains_pos(75));
        assert!(!single.contains_pos(25));
    }
    
    #[test]
    fn test_storages_equal() {
        let rs_factory = RangeSetStorageFactory::new(100, 100);
        let bdd_factory = BddStorageFactory::new(100, 100);
        
        let ranges: Vec<(usize, usize)> = vec![(10, 50), (100, 200), (500, 600)];
        
        let rs = rs_factory.from_ranges(ranges.clone().into_iter());
        let bdd = bdd_factory.from_ranges(ranges.into_iter());
        
        assert!(storages_equal(&rs, &bdd));
    }
    
    #[test]
    fn test_metrics() {
        let factory = BddStorageFactory::new(100, 100);
        
        let mut metrics = StorageMetrics::new("TestBDD");
        
        let s1 = factory.from_ranges(vec![(0, 100)].into_iter());
        let s2 = factory.from_ranges(vec![(200, 300), (500, 600)].into_iter());
        
        metrics.add(&s1);
        metrics.add(&s2);
        
        assert_eq!(metrics.num_weights, 2);
        assert!(metrics.total_bytes > 0);
    }
}
