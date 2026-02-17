//! Abstract weight type for DWA/NWA operations.
//! 
//! This module provides a unified weight type that can be used across
//! different weight representations. Currently only supports RangeSetBlaze,
//! but is designed to allow future extensions to other backends.
//!
//! # Adding a new backend
//! 1. Implement `WeightBackend` for your backend type.
//! 2. Add a new variant to `AbstractWeight` to wrap the backend.
//! 3. Update `AbstractWeight` constructors, accessors, and bitwise ops to
//!    dispatch to the new backend (mirroring the `RangeSet` variant).
//! 4. Add backend-specific tests that exercise the `WeightBackend` trait
//!    methods (ranges, set ops, complement, min/max, clip).
//!
//! # Backend selection
//! Use the `ABSTRACT_WEIGHT_BACKEND` environment variable to choose between
//! `rangeset` (default) and `factorized` backends.

use range_set_blaze::RangeSetBlaze;
use crate::datastructures::factorized_weight::{FactorizedWeight, intern_factorized};
use crate::datastructures::hybrid_bitset::RangeSet;
use crate::datastructures::rangemap_weight::{RangeMapWeight, intern_rangemap};
use crate::json_serialization::{JSONConvertible, JSONNode};
use profiler_macro::time_it;
use serde::{Deserialize, Serialize};
use serde::de::Error;
use serde_json::Value as JsonValue;
use std::cell::RefCell;
use std::hash::{Hash, Hasher};
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, Not};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

/// Dimensions for weight-heavy mode encoding.
/// 
/// In weight-heavy mode, weights are encoded in an N×M space where:
/// - N = number of LLM tokens (vocab_size)
/// - M = number of tokenizer states (num_tsids)
/// 
/// A position p in this space represents:
/// - Token ID: p / num_tsids
/// - Tokenizer state ID (tsid): p % num_tsids
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct WeightDimensions {
    /// Number of LLM tokens (vocab size). 0 for symbol-heavy mode.
    pub num_tokens: usize,
    /// Number of tokenizer states. 0 or 1 for symbol-heavy mode.
    pub num_tsids: usize,
}

impl WeightDimensions {
    /// Create new dimensions.
    pub fn new(num_tokens: usize, num_tsids: usize) -> Self {
        Self { num_tokens, num_tsids }
    }

    /// Check if this is weight-heavy mode (has tsid dimension).
    pub fn is_weight_heavy(&self) -> bool {
        self.num_tsids > 1
    }

    /// Total domain size: num_tokens × num_tsids.
    pub fn domain_size(&self) -> usize {
        if self.num_tsids == 0 {
            self.num_tokens
        } else {
            self.num_tokens * self.num_tsids
        }
    }

    /// Get the maximum position value (domain_size - 1).
    pub fn max_position(&self) -> usize {
        self.domain_size().saturating_sub(1)
    }

    /// Encode a token and tsid into a position.
    pub fn encode(&self, token: usize, tsid: usize) -> usize {
        debug_assert!(token < self.num_tokens, "token {} >= num_tokens {}", token, self.num_tokens);
        debug_assert!(tsid < self.num_tsids, "tsid {} >= num_tsids {}", tsid, self.num_tsids);
        token * self.num_tsids + tsid
    }

    /// Decode a position into (token, tsid).
    pub fn decode(&self, pos: usize) -> (usize, usize) {
        if self.num_tsids == 0 {
            (pos, 0)
        } else {
            (pos / self.num_tsids, pos % self.num_tsids)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendChoice {
    RangeSet,
    Factorized,
    RangeMap,
}

thread_local! {
    static BACKEND_OVERRIDE: RefCell<Option<BackendChoice>> = RefCell::new(None);
}

thread_local! {
    static ALLOW_EXPANSION_OVERRIDE: RefCell<Option<bool>> = RefCell::new(None);
}

thread_local! {
    static CACHED_EMPTY_WEIGHT: RefCell<Option<(BackendChoice, usize, AbstractWeight)>> = RefCell::new(None);
}

// --- AbstractWeight profiling ---
// Legacy weight-op profiling removed; keep no-op hooks for callers.
pub fn reset_weight_op_profiling() {}

pub fn print_weight_op_profiling(_label: &str) {}

#[derive(Clone, Default)]
pub(crate) struct BitorAssignCounters {
    pub(crate) owned_calls: u64,
    pub(crate) ref_calls: u64,
    pub(crate) rhs_empty: u64,
    pub(crate) self_empty: u64,
    pub(crate) self_all: u64,
    pub(crate) rhs_all: u64,
    pub(crate) union_total: u64,
    pub(crate) union_rangeset: u64,
    pub(crate) union_factorized: u64,
    pub(crate) union_rangemap: u64,
}

static BITOR_ASSIGN_OWNED_CALLS: AtomicU64 = AtomicU64::new(0);
static BITOR_ASSIGN_REF_CALLS: AtomicU64 = AtomicU64::new(0);
static BITOR_ASSIGN_RHS_EMPTY: AtomicU64 = AtomicU64::new(0);
static BITOR_ASSIGN_SELF_EMPTY: AtomicU64 = AtomicU64::new(0);
static BITOR_ASSIGN_SELF_ALL: AtomicU64 = AtomicU64::new(0);
static BITOR_ASSIGN_RHS_ALL: AtomicU64 = AtomicU64::new(0);
static BITOR_ASSIGN_UNION_TOTAL: AtomicU64 = AtomicU64::new(0);
static BITOR_ASSIGN_UNION_RANGESET: AtomicU64 = AtomicU64::new(0);
static BITOR_ASSIGN_UNION_FACTORIZED: AtomicU64 = AtomicU64::new(0);
static BITOR_ASSIGN_UNION_RANGEMAP: AtomicU64 = AtomicU64::new(0);

pub(crate) fn reset_bitor_assign_counters() {
    BITOR_ASSIGN_OWNED_CALLS.store(0, AtomicOrdering::Relaxed);
    BITOR_ASSIGN_REF_CALLS.store(0, AtomicOrdering::Relaxed);
    BITOR_ASSIGN_RHS_EMPTY.store(0, AtomicOrdering::Relaxed);
    BITOR_ASSIGN_SELF_EMPTY.store(0, AtomicOrdering::Relaxed);
    BITOR_ASSIGN_SELF_ALL.store(0, AtomicOrdering::Relaxed);
    BITOR_ASSIGN_RHS_ALL.store(0, AtomicOrdering::Relaxed);
    BITOR_ASSIGN_UNION_TOTAL.store(0, AtomicOrdering::Relaxed);
    BITOR_ASSIGN_UNION_RANGESET.store(0, AtomicOrdering::Relaxed);
    BITOR_ASSIGN_UNION_FACTORIZED.store(0, AtomicOrdering::Relaxed);
    BITOR_ASSIGN_UNION_RANGEMAP.store(0, AtomicOrdering::Relaxed);
}

pub(crate) fn bitor_assign_counters() -> BitorAssignCounters {
    BitorAssignCounters {
        owned_calls: BITOR_ASSIGN_OWNED_CALLS.load(AtomicOrdering::Relaxed),
        ref_calls: BITOR_ASSIGN_REF_CALLS.load(AtomicOrdering::Relaxed),
        rhs_empty: BITOR_ASSIGN_RHS_EMPTY.load(AtomicOrdering::Relaxed),
        self_empty: BITOR_ASSIGN_SELF_EMPTY.load(AtomicOrdering::Relaxed),
        self_all: BITOR_ASSIGN_SELF_ALL.load(AtomicOrdering::Relaxed),
        rhs_all: BITOR_ASSIGN_RHS_ALL.load(AtomicOrdering::Relaxed),
        union_total: BITOR_ASSIGN_UNION_TOTAL.load(AtomicOrdering::Relaxed),
        union_rangeset: BITOR_ASSIGN_UNION_RANGESET.load(AtomicOrdering::Relaxed),
        union_factorized: BITOR_ASSIGN_UNION_FACTORIZED.load(AtomicOrdering::Relaxed),
        union_rangemap: BITOR_ASSIGN_UNION_RANGEMAP.load(AtomicOrdering::Relaxed),
    }
}

pub(crate) fn bitor_assign_sample_should_check(counter: u64) -> bool {
    (counter & 1023) == 0
}

static BITOR_ASSIGN_NOOP_SAMPLE_TOTAL: AtomicU64 = AtomicU64::new(0);
static BITOR_ASSIGN_NOOP_SAMPLE_HIT: AtomicU64 = AtomicU64::new(0);

pub(crate) fn reset_bitor_assign_noop_sample() {
    BITOR_ASSIGN_NOOP_SAMPLE_TOTAL.store(0, AtomicOrdering::Relaxed);
    BITOR_ASSIGN_NOOP_SAMPLE_HIT.store(0, AtomicOrdering::Relaxed);
}

pub(crate) fn bitor_assign_noop_sample_counters() -> (u64, u64) {
    (
        BITOR_ASSIGN_NOOP_SAMPLE_TOTAL.load(AtomicOrdering::Relaxed),
        BITOR_ASSIGN_NOOP_SAMPLE_HIT.load(AtomicOrdering::Relaxed),
    )
}

/// Temporarily override the backend choice for the current thread.
/// Returns the previous value (if any) which should be restored later.
pub fn override_backend(choice: BackendChoice) -> Option<BackendChoice> {
    BACKEND_OVERRIDE.with(|cell| cell.borrow_mut().replace(choice))
}

/// Restore a previous backend override.
pub fn restore_backend(previous: Option<BackendChoice>) {
    BACKEND_OVERRIDE.with(|cell| *cell.borrow_mut() = previous);
}

/// Get the active backend choice for the current thread.
pub fn current_backend_choice() -> BackendChoice {
    backend_choice()
}

/// Check if factorized weight expansion is allowed for the current thread.
pub fn is_expansion_allowed() -> bool {
    ALLOW_EXPANSION_OVERRIDE.with(|cell| {
        if let Some(value) = *cell.borrow() {
            return value;
        }
        std::env::var("ALLOW_FACTORIZED_EXPANSION").is_ok()
    })
}

/// Override expansion allowance for the current thread.
pub fn override_expansion_allowed(allow: bool) -> Option<bool> {
    ALLOW_EXPANSION_OVERRIDE.with(|cell| cell.borrow_mut().replace(allow))
}

/// Restore previous expansion allowance override for the current thread.
pub fn restore_expansion_allowed(previous: Option<bool>) {
    ALLOW_EXPANSION_OVERRIDE.with(|cell| *cell.borrow_mut() = previous);
}

fn backend_choice() -> BackendChoice {
    if let Some(choice) = BACKEND_OVERRIDE.with(|cell| *cell.borrow()) {
        return choice;
    }
    match std::env::var("ABSTRACT_WEIGHT_BACKEND") {
        Ok(value) if value.eq_ignore_ascii_case("rsb") || value.eq_ignore_ascii_case("rangeset") => BackendChoice::RangeSet,
        Ok(value) if value.eq_ignore_ascii_case("factorized") => BackendChoice::Factorized,
        _ => BackendChoice::RangeMap, // Default to rangemap
    }
}

pub(crate) fn normalize_num_tsids(num_tsids: usize) -> usize {
    if num_tsids == 0 { 1 } else { num_tsids }
}

pub(crate) fn current_num_tsids() -> usize {
    normalize_num_tsids(crate::datastructures::get_num_tsids())
}

// ---------------------------------------------------------------------------
// WeightBackend Trait - defines all operations weight backends must implement
// ---------------------------------------------------------------------------

/// Trait defining the interface for weight backends.
/// 
/// Implementors of this trait provide the underlying representation and
/// operations for weights. Multiple backends can exist (e.g., RangeSetBlaze,
/// BDD, etc.) and the `AbstractWeight` enum dispatches to the appropriate
/// backend based on the variant.
pub trait WeightBackend: Clone + PartialEq + Eq + std::fmt::Debug {
    /// Create an empty weight.
    fn empty() -> Self;
    
    /// Create a weight containing all positions from 0 to max_position (inclusive).
    fn all(max_position: usize) -> Self;
    
    /// Create a weight from a single position.
    fn from_position(pos: usize) -> Self;
    
    /// Create a weight from an iterator of inclusive ranges.
    fn from_ranges<I: IntoIterator<Item = std::ops::RangeInclusive<usize>>>(ranges: I) -> Self;
    
    /// Check if the weight is empty.
    fn is_empty(&self) -> bool;
    
    /// Get the number of positions in the weight.
    fn len(&self) -> usize;
    
    /// Check if a position is in the weight.
    fn contains(&self, pos: usize) -> bool;
    
    /// Get the number of ranges in the weight.
    fn ranges_len(&self) -> usize;

    /// Get the number of ranges in the weight (backend-specific definition).
    fn num_ranges(&self) -> usize;
    
    /// Insert a position into the weight.
    fn insert(&mut self, pos: usize);
    
    /// Compute intersection with another weight.
    fn intersect(&self, other: &Self) -> Self;
    
    /// Compute intersection and mutate self.
    fn intersect_assign(&mut self, other: &Self);
    
    /// Compute union with another weight.
    fn union(&self, other: &Self) -> Self;
    
    /// Compute union and mutate self.
    fn union_assign(&mut self, other: &Self);
    
    /// Compute set difference (self - other).
    fn difference(&self, other: &Self) -> Self;
    
    /// Compute complement within the given max position (0..=max_position).
    fn complement(&self, max_position: usize) -> Self;
    
    /// Get the minimum position, if any.
    fn min_item(&self) -> Option<usize>;
    
    /// Get the maximum position, if any.
    fn max_item(&self) -> Option<usize>;
}

// ---------------------------------------------------------------------------
// RangeSet (HybridBitset) Backend Implementation
// ---------------------------------------------------------------------------

impl WeightBackend for RangeSet {
    fn empty() -> Self {
        RangeSet::zeros()
    }
    
    fn all(max_position: usize) -> Self {
        RangeSet::from(RangeSetBlaze::from_iter([0..=max_position]))
    }
    
    fn from_position(pos: usize) -> Self {
        RangeSet::from(RangeSetBlaze::from_iter([pos..=pos]))
    }
    
    fn from_ranges<I: IntoIterator<Item = std::ops::RangeInclusive<usize>>>(ranges: I) -> Self {
        RangeSet::from(RangeSetBlaze::from_iter(ranges))
    }
    
    fn is_empty(&self) -> bool {
        RangeSet::is_empty(self)
    }
    
    fn len(&self) -> usize {
        RangeSet::len(self)
    }
    
    fn contains(&self, pos: usize) -> bool {
        RangeSet::contains(self, pos)
    }
    
    fn ranges_len(&self) -> usize {
        RangeSet::ranges_len(self)
    }

    fn num_ranges(&self) -> usize {
        RangeSet::ranges_len(self)
    }
    
    fn insert(&mut self, pos: usize) {
        RangeSet::insert_with_intern(self, pos);
    }
    
    fn intersect(&self, other: &Self) -> Self {
        self & other
    }
    
    fn intersect_assign(&mut self, other: &Self) {
        *self = self.intersect(other);
    }
    
    fn union(&self, other: &Self) -> Self {
        self | other
    }
    
    fn union_assign(&mut self, other: &Self) {
        *self |= other;
    }
    
    fn difference(&self, other: &Self) -> Self {
        self - other
    }
    
    fn complement(&self, max_position: usize) -> Self {
        let all = RangeSet::from(RangeSetBlaze::from_iter([0..=max_position]));
        &all - self
    }
    
    fn min_item(&self) -> Option<usize> {
        self.ranges().next().map(|r| *r.start())
    }
    
    fn max_item(&self) -> Option<usize> {
        self.ranges().last().map(|r| *r.end())
    }
}


// ---------------------------------------------------------------------------
// AbstractWeight Enum - dispatches to backends
// ---------------------------------------------------------------------------

/// Abstract weight type wrapping different backends.
/// 
/// This enum provides a unified interface for weight operations.
/// Operations between different variants will panic - all weights in a
/// computation must use the same backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AbstractWeight {
    /// Weight represented as a RangeSet (HybridBitset).
    RangeSet(RangeSet),
    /// Weight represented as a factorized (tsid_set × token_set) union.
    Factorized(Arc<FactorizedWeight>),
    /// Weight represented as a RangeMapBlaze token->tsid mapping.
    RangeMap(Arc<RangeMapWeight>),
    // Future variants can be added here, e.g.:
    // Bdd(BddWeight),
    // Explicit(Vec<usize>),
}

impl std::hash::Hash for AbstractWeight {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            AbstractWeight::RangeSet(rsb) => {
                0u8.hash(state);
                for range in rsb.ranges() {
                    range.start().hash(state);
                    range.end().hash(state);
                }
            }
            AbstractWeight::Factorized(fw) => {
                1u8.hash(state);
                fw.hash(state);
            }
            AbstractWeight::RangeMap(rm) => {
                2u8.hash(state);
                rm.hash(state);
            }
        }
    }
}

impl PartialOrd for AbstractWeight {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        // Compare by len, then by first range
        match self.len().cmp(&other.len()) {
            std::cmp::Ordering::Equal => {
                let self_first = self.min_item();
                let other_first = other.min_item();
                self_first.partial_cmp(&other_first)
            }
            ord => Some(ord),
        }
    }
}

impl Ord for AbstractWeight {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl std::fmt::Display for AbstractWeight {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AbstractWeight::RangeSet(rsb) => {
                let ranges: Vec<_> = rsb.ranges().take(5).collect();
                if ranges.len() < rsb.ranges_len() {
                    write!(f, "Weight({} ranges, {} items)", rsb.ranges_len(), self.len())
                } else {
                    write!(f, "Weight({:?})", ranges)
                }
            }
            AbstractWeight::Factorized(fw) => {
                write!(f, "FactorizedWeight({} pairs)", fw.pairs.len())
            }
            AbstractWeight::RangeMap(rm) => {
                write!(f, "RangeMapWeight({} ranges)", rm.map.range_values().count())
            }
        }
    }
}

/// Serde proxy enum for bincode-compatible serialization of AbstractWeight.
/// Preserves the variant discriminant so all weight types round-trip correctly.
#[derive(serde::Serialize, serde::Deserialize)]
enum AbstractWeightProxy {
    RangeSet(RangeSet),
    Factorized(FactorizedWeight),
    RangeMap(RangeMapWeight),
}

impl Serialize for AbstractWeight {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let proxy = match self {
            AbstractWeight::RangeSet(rs) => AbstractWeightProxy::RangeSet(rs.clone()),
            AbstractWeight::Factorized(fw) => AbstractWeightProxy::Factorized((**fw).clone()),
            AbstractWeight::RangeMap(rm) => AbstractWeightProxy::RangeMap((**rm).clone()),
        };
        proxy.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for AbstractWeight {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let proxy = AbstractWeightProxy::deserialize(deserializer)?;
        Ok(match proxy {
            AbstractWeightProxy::RangeSet(rs) => AbstractWeight::RangeSet(rs),
            AbstractWeightProxy::Factorized(fw) => AbstractWeight::Factorized(Arc::new(fw)),
            AbstractWeightProxy::RangeMap(rm) => AbstractWeight::RangeMap(Arc::new(rm)),
        })
    }
}

impl JSONConvertible for AbstractWeight {
    fn to_json(&self) -> JSONNode {
        match self {
            AbstractWeight::RangeSet(rsb) => {
                // Serialize as ranges: [[start, end], ...]
                let ranges_vec: Vec<Vec<usize>> = rsb
                    .ranges()
                    .map(|ri| vec![*ri.start(), *ri.end()])
                    .collect();
                let mut obj = std::collections::BTreeMap::new();
                obj.insert("type".to_string(), JSONNode::String("rangeset".to_string()));
                obj.insert("ranges".to_string(), ranges_vec.to_json());
                JSONNode::Object(obj)
            }
            AbstractWeight::Factorized(fw) => {
                // Serialize factorized representation: pairs of (tsid_set, token_set)
                let pairs: Vec<(Vec<Vec<usize>>, Vec<Vec<usize>>)> = fw.pairs()
                    .iter()
                    .map(|(tsid_set, token_set)| {
                        let tsid_ranges: Vec<Vec<usize>> = tsid_set.ranges()
                            .map(|ri| vec![*ri.start(), *ri.end()])
                            .collect();
                        let token_ranges: Vec<Vec<usize>> = token_set.ranges()
                            .map(|ri| vec![*ri.start(), *ri.end()])
                            .collect();
                        (tsid_ranges, token_ranges)
                    })
                    .collect();
                let mut obj = std::collections::BTreeMap::new();
                obj.insert("type".to_string(), JSONNode::String("factorized".to_string()));
                obj.insert("num_tsids".to_string(), JSONNode::UInt(fw.num_tsids() as u128));
                obj.insert("pairs".to_string(), pairs.to_json());
                JSONNode::Object(obj)
            }
            AbstractWeight::RangeMap(rm) => {
                let ranges_vec: Vec<Vec<usize>> = rm
                    .expand_to_rsb()
                    .ranges()
                    .map(|ri| vec![*ri.start(), *ri.end()])
                    .collect();
                let mut obj = std::collections::BTreeMap::new();
                obj.insert("type".to_string(), JSONNode::String("rangemap".to_string()));
                obj.insert("num_tsids".to_string(), JSONNode::UInt(rm.num_tsids() as u128));
                obj.insert("ranges".to_string(), ranges_vec.to_json());
                JSONNode::Object(obj)
            }
        }
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        // Try to parse as new format (with "type" field)
        if let Ok(mut obj) = node.clone().into_object() {
            if let Some(type_node) = obj.remove("type") {
                let type_str = match type_node {
                    JSONNode::String(s) => s,
                    other => return Err(format!("Expected string for type, got {:?}", other)),
                };
                match type_str.as_str() {
                    "rangeset" => {
                        let ranges_vec: Vec<Vec<usize>> = Vec::from_json(
                            obj.remove("ranges").ok_or("Missing ranges")?
                        )?;
                        let mut ranges = Vec::new();
                        for mut v in ranges_vec {
                            if v.len() != 2 {
                                return Err(format!("Expected 2-element array, got {:?}", v));
                            }
                            let end = v.pop().unwrap();
                            let start = v.pop().unwrap();
                            ranges.push(start..=end);
                        }
                        return Ok(AbstractWeight::from_rsb(RangeSetBlaze::from_iter(ranges)));
                    }
                    "factorized" => {
                        let num_tsids: usize = match obj.remove("num_tsids") {
                            Some(JSONNode::UInt(n)) => n as usize,
                            Some(JSONNode::Int(n)) => n as usize,
                            _ => return Err("Missing or invalid num_tsids".to_string()),
                        };
                        let pairs_json = obj.remove("pairs").ok_or("Missing pairs")?;
                        let pairs_vec: Vec<(Vec<Vec<usize>>, Vec<Vec<usize>>)> = Vec::from_json(pairs_json)?;
                        let pairs: Vec<(RangeSet, RangeSet)> = pairs_vec
                            .into_iter()
                            .map(|(tsid_ranges, token_ranges)| {
                                let tsid_set = RangeSet::from(RangeSetBlaze::from_iter(
                                    tsid_ranges.into_iter().map(|v| v[0]..=v[1])
                                ));
                                let token_set = RangeSet::from(RangeSetBlaze::from_iter(
                                    token_ranges.into_iter().map(|v| v[0]..=v[1])
                                ));
                                (tsid_set, token_set)
                            })
                            .collect();
                        let fw = FactorizedWeight::from_pairs(pairs, num_tsids);
                        return Ok(match backend_choice() {
                            BackendChoice::RangeSet => {
                                AbstractWeight::from_rsb(fw.expand_to_rsb_unchecked())
                            }
                            BackendChoice::Factorized => AbstractWeight::Factorized(intern_factorized(fw)),
                            BackendChoice::RangeMap => {
                                AbstractWeight::RangeMap(intern_rangemap(
                                    RangeMapWeight::from_rsb_with_num_tsids(
                                        &fw.expand_to_rsb_unchecked(),
                                        num_tsids,
                                    ),
                                ))
                            }
                        });
                    }
                    "rangemap" => {
                        let ranges_vec: Vec<Vec<usize>> = Vec::from_json(
                            obj.remove("ranges").ok_or("Missing ranges")?
                        )?;
                        let mut ranges = Vec::new();
                        for mut v in ranges_vec {
                            if v.len() != 2 {
                                return Err(format!("Expected 2-element array, got {:?}", v));
                            }
                            let end = v.pop().unwrap();
                            let start = v.pop().unwrap();
                            ranges.push(start..=end);
                        }
                        return Ok(AbstractWeight::from_rsb(RangeSetBlaze::from_iter(ranges)));
                    }
                    _ => return Err(format!("Unknown weight type: {}", type_str)),
                }
            }
        }
        
        // Fall back to old format (just ranges array) for backward compatibility
        let rsb = crate::datastructures::hybrid_bitset::RangeSet::from_json(node)?;
        Ok(AbstractWeight::from_rsb(std::sync::Arc::unwrap_or_clone(rsb.inner)))
    }
}

impl std::ops::Not for AbstractWeight {
    type Output = AbstractWeight;
    
    #[time_it("AbstractWeight::not")]
    fn not(self) -> Self::Output {
        self.complement()
    }
}

impl std::ops::Not for &AbstractWeight {
    type Output = AbstractWeight;
    
    fn not(self) -> Self::Output {
        self.complement()
    }
}

impl Default for AbstractWeight {
    fn default() -> Self {
        AbstractWeight::empty()
    }
}

impl AbstractWeight {
    fn empty_uncached() -> Self {
        match backend_choice() {
            BackendChoice::RangeSet => AbstractWeight::RangeSet(RangeSet::zeros()),
            BackendChoice::Factorized => {
                AbstractWeight::Factorized(intern_factorized(FactorizedWeight::new(current_num_tsids())))
            }
            BackendChoice::RangeMap => {
                AbstractWeight::RangeMap(intern_rangemap(RangeMapWeight::new(current_num_tsids())))
            }
        }
    }

    /// Create an empty weight (no positions).
    pub fn empty() -> Self {
        let choice = backend_choice();
        let num_tsids = current_num_tsids();
        CACHED_EMPTY_WEIGHT.with(|cell| {
            let cached = cell
                .borrow()
                .as_ref()
                .and_then(|(cached_choice, cached_tsids, cached_weight)| {
                    if *cached_choice == choice && *cached_tsids == num_tsids {
                        Some(cached_weight.clone())
                    } else {
                        None
                    }
                });
            if let Some(weight) = cached {
                return weight;
            }
            let weight = Self::empty_uncached();
            *cell.borrow_mut() = Some((choice, num_tsids, weight.clone()));
            weight
        })
    }
    
    /// Create an empty weight (alias for `empty()`).
    pub fn zeros() -> Self {
        Self::empty()
    }
    
    /// Create a weight containing all positions.
    /// 
    /// Uses the global dims (from set_global_dims) to determine the domain.
    /// This is the preferred no-arg version.
    pub fn ones() -> Self {
        let max_llm_token = crate::datastructures::get_max_llm_token();
        let num_tsids = crate::datastructures::get_num_tsids();
        let domain_max = if num_tsids > 1 {
            max_llm_token
                .saturating_mul(num_tsids)
                .saturating_add(num_tsids.saturating_sub(1))
        } else {
            max_llm_token
        };
        match backend_choice() {
            BackendChoice::RangeSet => {
                AbstractWeight::RangeSet(RangeSet::from(RangeSetBlaze::from_iter([0..=domain_max])))
            }
            BackendChoice::Factorized => AbstractWeight::Factorized(
                intern_factorized(FactorizedWeight::all_with_max_position(
                    domain_max,
                    normalize_num_tsids(num_tsids),
                )),
            ),
            BackendChoice::RangeMap => AbstractWeight::RangeMap(
                intern_rangemap(<RangeMapWeight as WeightBackend>::all(domain_max)),
            ),
        }
    }
    
    /// Create a weight containing all positions (alias for `ones()`).
    /// 
    /// Uses the global dims (from set_global_dims) to determine the domain.
    pub fn all() -> Self {
        Self::ones()
    }

    /// Create a weight containing all positions in the given range.
    pub fn all_with_dims(dims: WeightDimensions) -> Self {
        if dims.domain_size() == 0 {
            return Self::empty();
        }
        match backend_choice() {
            BackendChoice::RangeSet => {
                AbstractWeight::RangeSet(RangeSet::from(RangeSetBlaze::from_iter([0..=dims.max_position()])))
            }
            BackendChoice::Factorized => AbstractWeight::Factorized(
                intern_factorized(FactorizedWeight::all_with_max_position(
                    dims.max_position(),
                    normalize_num_tsids(dims.num_tsids),
                )),
            ),
            BackendChoice::RangeMap => AbstractWeight::RangeMap(
                intern_rangemap(<RangeMapWeight as WeightBackend>::all(dims.max_position())),
            ),
        }
    }

    /// Create a weight from a single position.
    pub fn from_position(pos: usize) -> Self {
        match backend_choice() {
            BackendChoice::RangeSet => {
                AbstractWeight::RangeSet(RangeSet::from_item(pos))
            }
            BackendChoice::Factorized => AbstractWeight::Factorized(
                intern_factorized(FactorizedWeight::from_position_with_num_tsids(pos, current_num_tsids())),
            ),
            BackendChoice::RangeMap => AbstractWeight::RangeMap(
                intern_rangemap(<RangeMapWeight as WeightBackend>::from_position(pos)),
            ),
        }
    }

    /// Create a weight from a single position (alias for `from_position`).
    pub fn from_item(pos: usize) -> Self {
        Self::from_position(pos)
    }

    /// Create a weight from an iterator of positions.
    pub fn from_iter<I: IntoIterator<Item = usize>>(iter: I) -> Self {
        AbstractWeight::from_rsb(RangeSetBlaze::from_iter(iter))
    }
    
    /// Create a weight from an iterator of inclusive ranges.
    pub fn from_ranges<I: IntoIterator<Item = std::ops::RangeInclusive<usize>>>(ranges: I) -> Self {
        AbstractWeight::from_rsb(RangeSetBlaze::<usize>::from_iter(ranges))
    }

    /// Create a weight from a RangeSetBlaze.
    pub fn from_rsb(rsb: RangeSetBlaze<usize>) -> Self {
        match backend_choice() {
            BackendChoice::RangeSet => AbstractWeight::RangeSet(RangeSet::from(rsb)),
            BackendChoice::Factorized => AbstractWeight::Factorized(
                intern_factorized(FactorizedWeight::from_rsb_with_num_tsids(&rsb, current_num_tsids())),
            ),
            BackendChoice::RangeMap => AbstractWeight::RangeMap(
                intern_rangemap(RangeMapWeight::from_rsb_with_num_tsids(&rsb, current_num_tsids())),
            ),
        }
    }

    /// Union multiple weights in a single operation.
    // #[time_it]
    pub fn bulk_union(weights: &[&AbstractWeight]) -> AbstractWeight {
        match weights.len() {
            0 => AbstractWeight::zeros(),
            1 => weights[0].clone(),
            2 => {
                let mut result = weights[0].clone();
                result |= weights[1];
                result
            }
            3 => {
                let mut result = weights[0].clone();
                result |= weights[1];
                result |= weights[2];
                result
            }
            4 => {
                let mut left = weights[0].clone();
                left |= weights[1];
                let mut right = weights[2].clone();
                right |= weights[3];
                left |= &right;
                left
            }
            _ => match weights[0] {
                AbstractWeight::RangeSet(_) => {
                    let mut sets: Vec<&RangeSet> = Vec::with_capacity(weights.len());
                    for weight in weights {
                        if let AbstractWeight::RangeSet(rsb) = weight {
                            sets.push(rsb);
                        } else {
                            panic!("AbstractWeight::bulk_union requires all weights to be the same variant");
                        }
                    }
                    AbstractWeight::RangeSet(RangeSet::bulk_union(&sets))
                }
                AbstractWeight::Factorized(_) => {
                    let mut out = weights[0].clone();
                    for weight in &weights[1..] {
                        out |= *weight;
                    }
                    out
                }
                AbstractWeight::RangeMap(_) => {
                    let mut maps: Vec<&RangeMapWeight> = Vec::with_capacity(weights.len());
                    for weight in weights {
                        if let AbstractWeight::RangeMap(rm) = weight {
                            maps.push(rm.as_ref());
                        } else {
                            panic!("AbstractWeight::bulk_union requires all weights to be the same variant");
                        }
                    }
                    AbstractWeight::RangeMap(RangeMapWeight::bulk_union(&maps))
                }
            },
        }
    }


    /// Check if the weight is empty.
    pub fn is_empty(&self) -> bool {
        match self {
            AbstractWeight::RangeSet(rsb) => WeightBackend::is_empty(rsb),
            AbstractWeight::Factorized(fw) => WeightBackend::is_empty(fw),
            AbstractWeight::RangeMap(rm) => WeightBackend::is_empty(rm),
        }
    }

    /// Check if two weights are disjoint (no overlapping positions).
    /// This is cheaper than computing the full intersection and checking is_empty,
    /// because it can early-exit as soon as any overlap is found.
    pub fn is_disjoint_with(&self, other: &Self) -> bool {
        if self.is_empty() || other.is_empty() {
            return true;
        }
        match (self, other) {
            (AbstractWeight::RangeSet(a), AbstractWeight::RangeSet(b)) => {
                a.is_disjoint(b)
            }
            (AbstractWeight::RangeMap(a), AbstractWeight::RangeMap(b)) => {
                a.is_disjoint(b)
            }
            _ => {
                // Fallback: compute intersection and check empty
                (self & other).is_empty()
            }
        }
    }

    /// Get the number of positions in the weight.
    pub fn len(&self) -> usize {
        match self {
            AbstractWeight::RangeSet(rsb) => WeightBackend::len(rsb),
            AbstractWeight::Factorized(fw) => WeightBackend::len(fw),
            AbstractWeight::RangeMap(rm) => WeightBackend::len(rm),
        }
    }

    /// Check if a position is in the weight.
    pub fn contains(&self, pos: usize) -> bool {
        match self {
            AbstractWeight::RangeSet(rsb) => WeightBackend::contains(rsb, pos),
            AbstractWeight::Factorized(fw) => WeightBackend::contains(fw, pos),
            AbstractWeight::RangeMap(rm) => WeightBackend::contains(rm, pos),
        }
    }

    /// Expand to a RangeSetBlaze representation.
    pub fn to_rsb(&self) -> RangeSetBlaze<usize> {
        match self {
            AbstractWeight::RangeSet(rsb) => rsb.inner().clone(),
            AbstractWeight::Factorized(fw) => {
                if !is_expansion_allowed() {
                    panic!(
                        "Unexpected factorized weight expansion at: AbstractWeight::to_rsb(). Set ALLOW_FACTORIZED_EXPANSION=1 to allow."
                    );
                }
                fw.expand_to_rsb()
            }
            AbstractWeight::RangeMap(rm) => rm.expand_to_rsb(),
        }
    }

    /// Expand to a RangeSetBlaze representation, allowing expansion.
    ///
    /// Use sparingly in runtime paths where expansion is acceptable.
    pub fn to_rsb_allow_expansion(&self) -> RangeSetBlaze<usize> {
        match self {
            AbstractWeight::RangeSet(rsb) => rsb.inner().clone(),
            AbstractWeight::Factorized(fw) => fw.expand_to_rsb_unchecked(),
            AbstractWeight::RangeMap(rm) => rm.expand_to_rsb(),
        }
    }

    /// Cached version of `to_rsb_allow_expansion()`.
    ///
    /// For Factorized and RangeMap weights (which are Arc-wrapped and immutable),
    /// the expansion result is cached in a thread-local HashMap keyed by Arc pointer
    /// identity. This avoids re-expanding the same DWA transition weights on every
    /// mask generation step.
    ///
    /// Returns `Arc<RangeSetBlaze<usize>>` for cheap cloning.
    pub fn to_rsb_allow_expansion_cached(&self) -> Arc<RangeSetBlaze<usize>> {
        thread_local! {
            static CACHE: RefCell<std::collections::HashMap<usize, Arc<RangeSetBlaze<usize>>>> =
                RefCell::new(std::collections::HashMap::new());
        }

        match self {
            AbstractWeight::RangeSet(rsb) => {
                // RangeSet is already a thin wrapper; no expensive expansion.
                Arc::new(rsb.inner().clone())
            }
            AbstractWeight::Factorized(arc) => {
                let key = Arc::as_ptr(arc) as usize;
                CACHE.with(|cache| {
                    let mut c = cache.borrow_mut();
                    c.entry(key)
                        .or_insert_with(|| Arc::new(arc.expand_to_rsb_unchecked()))
                        .clone()
                })
            }
            AbstractWeight::RangeMap(arc) => {
                let key = Arc::as_ptr(arc) as usize;
                CACHE.with(|cache| {
                    let mut c = cache.borrow_mut();
                    c.entry(key)
                        .or_insert_with(|| Arc::new(arc.expand_to_rsb()))
                        .clone()
                })
            }
        }
    }

    /// Expand to a RangeSetBlaze representation (alias for `to_rsb`).
    pub fn expand_to_rsb(&self) -> RangeSetBlaze<usize> {
        self.to_rsb()
    }

    /// Get the number of ranges in the weight.
    pub fn ranges_len(&self) -> usize {
        match self {
            AbstractWeight::RangeSet(rsb) => WeightBackend::ranges_len(rsb),
            AbstractWeight::Factorized(fw) => WeightBackend::ranges_len(fw),
            AbstractWeight::RangeMap(rm) => WeightBackend::ranges_len(rm),
        }
    }
    
    /// Alias for `ranges_len()`.
    pub fn num_ranges(&self) -> usize {
        match self {
            AbstractWeight::RangeSet(rsb) => WeightBackend::num_ranges(rsb),
            AbstractWeight::Factorized(fw) => WeightBackend::num_ranges(fw),
            AbstractWeight::RangeMap(rm) => WeightBackend::num_ranges(rm),
        }
    }
    
    /// Fast check if weight represents all positions in the domain.
    /// 
    /// Uses global dims to determine domain size.
    pub fn is_all_fast(&self) -> bool {
        let max_llm_token = crate::datastructures::get_max_llm_token();
        let num_tsids = normalize_num_tsids(crate::datastructures::get_num_tsids());
        match self {
            AbstractWeight::RangeMap(rm) => {
                let mut ranges = rm.map.range_values();
                if ranges.len() != 1 {
                    return false;
                }
                let (token_range, tsid_set) = match ranges.next() {
                    Some(value) => value,
                    None => return false,
                };
                if *token_range.start() != 0 || *token_range.end() != max_llm_token {
                    return false;
                }
                if tsid_set.ranges_len() != 1 {
                    return false;
                }
                let mut tsid_ranges = tsid_set.ranges();
                let tsid_range = match tsid_ranges.next() {
                    Some(value) => value,
                    None => return false,
                };
                *tsid_range.start() == 0 && *tsid_range.end() == num_tsids.saturating_sub(1)
            }
            _ => {
                if self.ranges_len() != 1 {
                    return false;
                }
                let domain_size = max_llm_token
                    .saturating_add(1)
                    .saturating_mul(num_tsids);
                self.len() == domain_size
            }
        }
    }
    
    /// Check if self is a subset of other.
    pub fn is_subset_of(&self, other: &Self) -> bool {
        if self.is_empty() {
            return true;
        }
        if other.is_empty() {
            return false;
        }
        match (self, other) {
            (AbstractWeight::RangeMap(a), AbstractWeight::RangeMap(b)) => {
                a.is_subset_of(b)
            }
            _ => {
                // Fallback: compute difference and check empty
                self.difference(other).is_empty()
            }
        }
    }

    /// Iterate over ranges.
    pub fn ranges(&self) -> Box<dyn Iterator<Item = std::ops::RangeInclusive<usize>> + '_> {
        match self {
            AbstractWeight::RangeSet(rsb) => Box::new(rsb.ranges()),
            AbstractWeight::Factorized(fw) => {
                let ranges: Vec<_> = fw.expand_to_rsb().ranges().collect();
                Box::new(ranges.into_iter())
            }
            AbstractWeight::RangeMap(rm) => {
                let ranges: Vec<_> = rm.expand_to_rsb().ranges().collect();
                Box::new(ranges.into_iter())
            }
        }
    }

    /// Insert a position.
    pub fn insert(&mut self, pos: usize) {
        match self {
            AbstractWeight::RangeSet(rsb) => WeightBackend::insert(rsb, pos),
            AbstractWeight::Factorized(fw) => WeightBackend::insert(fw, pos),
            AbstractWeight::RangeMap(rm) => WeightBackend::insert(rm, pos),
        }
    }

    /// Clip this weight to the range 0..=max without building a large clip weight.
    pub fn clip_to_max(&mut self, max: usize) {
        match self {
            AbstractWeight::RangeSet(rsb) => {
                let clip = RangeSet::from(RangeSetBlaze::from_iter([0..=max]));
                *rsb &= &clip;
            }
            AbstractWeight::Factorized(fw) => {
                let num_tsids = fw.num_tsids();
                let clip = intern_factorized(FactorizedWeight::all_with_max_position(max, num_tsids));
                *fw = WeightBackend::intersect(fw, &clip);
            }
            AbstractWeight::RangeMap(rm) => {
                let mut new = (**rm).clone();
                new.clip_to_max(max);
                *rm = intern_rangemap(new);
            }
        }
    }

    /// Remove a position.
    pub fn remove(&mut self, pos: usize) {
        let single = AbstractWeight::from_position(pos);
        *self = self.difference(&single);
    }

    /// Set or clear a position.
    pub fn set(&mut self, pos: usize, value: bool) {
        if value {
            self.insert(pos);
        } else {
            self.remove(pos);
        }
    }
    
    /// Get the minimum position, if any.
    pub fn min_item(&self) -> Option<usize> {
        match self {
            AbstractWeight::RangeSet(rsb) => WeightBackend::min_item(rsb),
            AbstractWeight::Factorized(fw) => WeightBackend::min_item(fw),
            AbstractWeight::RangeMap(rm) => WeightBackend::min_item(rm),
        }
    }
    
    /// Get the maximum position, if any.
    pub fn max_item(&self) -> Option<usize> {
        match self {
            AbstractWeight::RangeSet(rsb) => WeightBackend::max_item(rsb),
            AbstractWeight::Factorized(fw) => WeightBackend::max_item(fw),
            AbstractWeight::RangeMap(rm) => WeightBackend::max_item(rm),
        }
    }

    /// Compute set difference (self - other).
    /// 
    /// Panics if self and other are different variants.
    pub fn difference(&self, other: &Self) -> Self {
        match (self, other) {
            (AbstractWeight::RangeSet(a), AbstractWeight::RangeSet(b)) => {
                AbstractWeight::RangeSet(WeightBackend::difference(a, b))
            }
            (AbstractWeight::Factorized(a), AbstractWeight::Factorized(b)) => {
                AbstractWeight::Factorized(WeightBackend::difference(a, b))
            }
            (AbstractWeight::RangeMap(a), AbstractWeight::RangeMap(b)) => {
                AbstractWeight::RangeMap(WeightBackend::difference(a, b))
            }
            _ => panic!("AbstractWeight operation requires both operands to be the same variant"),
        }
    }

    /// Compute semiring divide (self ∪ complement(other)).
    pub fn divide(&self, other: &Self) -> Self {
        let result = match (self, other) {
            (AbstractWeight::RangeMap(a), AbstractWeight::RangeMap(b)) => {
                // Use separate cached divide operation
                AbstractWeight::RangeMap(crate::datastructures::rangemap_weight::divide_rangemap_cached(a, b))
            }
            (AbstractWeight::RangeSet(_), AbstractWeight::RangeSet(_)) => {
                self | &other.complement()
            }
            (AbstractWeight::Factorized(_), AbstractWeight::Factorized(_)) => {
                self | &other.complement()
            }
            _ => panic!("AbstractWeight operation requires both operands to be the same variant"),
        };
        result
    }
    
}

// ---------------------------------------------------------------------------
// Bitwise Operations - dispatch to backends with variant checking
// ---------------------------------------------------------------------------

impl BitAnd for AbstractWeight {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (AbstractWeight::RangeSet(a), AbstractWeight::RangeSet(b)) => {
                AbstractWeight::RangeSet(WeightBackend::intersect(&a, &b))
            }
            (AbstractWeight::Factorized(a), AbstractWeight::Factorized(b)) => {
                AbstractWeight::Factorized(WeightBackend::intersect(&a, &b))
            }
            (AbstractWeight::RangeMap(a), AbstractWeight::RangeMap(b)) => {
                AbstractWeight::RangeMap(WeightBackend::intersect(&a, &b))
            }
            _ => panic!("AbstractWeight operation requires both operands to be the same variant"),
        }
    }
}

impl BitAnd<&AbstractWeight> for AbstractWeight {
    type Output = AbstractWeight;

    fn bitand(self, rhs: &AbstractWeight) -> Self::Output {
        match (self, rhs) {
            (AbstractWeight::RangeSet(a), AbstractWeight::RangeSet(b)) => {
                AbstractWeight::RangeSet(WeightBackend::intersect(&a, b))
            }
            (AbstractWeight::Factorized(a), AbstractWeight::Factorized(b)) => {
                AbstractWeight::Factorized(WeightBackend::intersect(&a, b))
            }
            (AbstractWeight::RangeMap(a), AbstractWeight::RangeMap(b)) => {
                AbstractWeight::RangeMap(WeightBackend::intersect(&a, b))
            }
            _ => panic!("AbstractWeight operation requires both operands to be the same variant"),
        }
    }
}

impl BitAnd for &AbstractWeight {
    type Output = AbstractWeight;

    fn bitand(self, rhs: Self) -> Self::Output {
        if self.is_empty() || rhs.is_empty() {
            return AbstractWeight::zeros();
        }
        if self.is_all_fast() {
            return rhs.clone();
        }
        if rhs.is_all_fast() {
            return self.clone();
        }
        match (self, rhs) {
            (AbstractWeight::RangeSet(a), AbstractWeight::RangeSet(b)) => {
                AbstractWeight::RangeSet(WeightBackend::intersect(a, b))
            }
            (AbstractWeight::Factorized(a), AbstractWeight::Factorized(b)) => {
                AbstractWeight::Factorized(WeightBackend::intersect(a, b))
            }
            (AbstractWeight::RangeMap(a), AbstractWeight::RangeMap(b)) => {
                AbstractWeight::RangeMap(WeightBackend::intersect(a, b))
            }
            _ => panic!("AbstractWeight operation requires both operands to be the same variant"),
        }
    }
}

impl BitAndAssign for AbstractWeight {
    fn bitand_assign(&mut self, rhs: Self) {
        match (self, &rhs) {
            (AbstractWeight::RangeSet(a), AbstractWeight::RangeSet(b)) => {
                WeightBackend::intersect_assign(a, b);
            }
            (AbstractWeight::Factorized(a), AbstractWeight::Factorized(b)) => {
                WeightBackend::intersect_assign(a, b);
            }
            (AbstractWeight::RangeMap(a), AbstractWeight::RangeMap(b)) => {
                WeightBackend::intersect_assign(a, b);
            }
            _ => panic!("AbstractWeight operation requires both operands to be the same variant"),
        }
    }
}

impl BitAndAssign<&AbstractWeight> for AbstractWeight {
    fn bitand_assign(&mut self, rhs: &AbstractWeight) {
        match (self, rhs) {
            (AbstractWeight::RangeSet(a), AbstractWeight::RangeSet(b)) => {
                WeightBackend::intersect_assign(a, b);
            }
            (AbstractWeight::Factorized(a), AbstractWeight::Factorized(b)) => {
                WeightBackend::intersect_assign(a, b);
            }
            (AbstractWeight::RangeMap(a), AbstractWeight::RangeMap(b)) => {
                WeightBackend::intersect_assign(a, b);
            }
            _ => panic!("AbstractWeight operation requires both operands to be the same variant"),
        }
    }
}

impl BitOr for AbstractWeight {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (AbstractWeight::RangeSet(a), AbstractWeight::RangeSet(b)) => {
                AbstractWeight::RangeSet(WeightBackend::union(&a, &b))
            }
            (AbstractWeight::Factorized(a), AbstractWeight::Factorized(b)) => {
                AbstractWeight::Factorized(WeightBackend::union(&a, &b))
            }
            (AbstractWeight::RangeMap(a), AbstractWeight::RangeMap(b)) => {
                AbstractWeight::RangeMap(WeightBackend::union(&a, &b))
            }
            _ => panic!("AbstractWeight operation requires both operands to be the same variant"),
        }
    }
}

impl BitOr<&AbstractWeight> for AbstractWeight {
    type Output = AbstractWeight;

    fn bitor(self, rhs: &AbstractWeight) -> Self::Output {
        if std::mem::discriminant(&self) != std::mem::discriminant(rhs) {
            panic!("AbstractWeight operation requires both operands to be the same variant");
        }
        if rhs.is_empty() {
            return self;
        }
        if self.is_empty() {
            return rhs.clone();
        }
        if self.is_all_fast() {
            return self;
        }
        if rhs.is_all_fast() {
            return rhs.clone();
        }
        match (self, rhs) {
            (AbstractWeight::RangeSet(a), AbstractWeight::RangeSet(b)) => {
                AbstractWeight::RangeSet(WeightBackend::union(&a, b))
            }
            (AbstractWeight::Factorized(a), AbstractWeight::Factorized(b)) => {
                AbstractWeight::Factorized(WeightBackend::union(&a, b))
            }
            (AbstractWeight::RangeMap(a), AbstractWeight::RangeMap(b)) => {
                AbstractWeight::RangeMap(WeightBackend::union(&a, b))
            }
            _ => panic!("AbstractWeight operation requires both operands to be the same variant"),
        }
    }
}

impl BitOr for &AbstractWeight {
    type Output = AbstractWeight;

    fn bitor(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (AbstractWeight::RangeSet(a), AbstractWeight::RangeSet(b)) => {
                AbstractWeight::RangeSet(WeightBackend::union(a, b))
            }
            (AbstractWeight::Factorized(a), AbstractWeight::Factorized(b)) => {
                AbstractWeight::Factorized(WeightBackend::union(a, b))
            }
            (AbstractWeight::RangeMap(a), AbstractWeight::RangeMap(b)) => {
                AbstractWeight::RangeMap(WeightBackend::union(a, b))
            }
            _ => panic!("AbstractWeight operation requires both operands to be the same variant"),
        }
    }
}

impl BitOrAssign for AbstractWeight {
    fn bitor_assign(&mut self, rhs: Self) {
        BITOR_ASSIGN_OWNED_CALLS.fetch_add(1, AtomicOrdering::Relaxed);
        match (self, &rhs) {
            (AbstractWeight::RangeSet(a), AbstractWeight::RangeSet(b)) => {
                BITOR_ASSIGN_UNION_TOTAL.fetch_add(1, AtomicOrdering::Relaxed);
                BITOR_ASSIGN_UNION_RANGESET.fetch_add(1, AtomicOrdering::Relaxed);
                WeightBackend::union_assign(a, b);
            }
            (AbstractWeight::Factorized(a), AbstractWeight::Factorized(b)) => {
                BITOR_ASSIGN_UNION_TOTAL.fetch_add(1, AtomicOrdering::Relaxed);
                BITOR_ASSIGN_UNION_FACTORIZED.fetch_add(1, AtomicOrdering::Relaxed);
                WeightBackend::union_assign(a, b);
            }
            (AbstractWeight::RangeMap(a), AbstractWeight::RangeMap(b)) => {
                BITOR_ASSIGN_UNION_TOTAL.fetch_add(1, AtomicOrdering::Relaxed);
                BITOR_ASSIGN_UNION_RANGEMAP.fetch_add(1, AtomicOrdering::Relaxed);
                WeightBackend::union_assign(a, b);
            }
            _ => panic!("AbstractWeight operation requires both operands to be the same variant"),
        }
    }
}

impl BitOrAssign<&AbstractWeight> for AbstractWeight {
    fn bitor_assign(&mut self, rhs: &AbstractWeight) {
        let call_idx = BITOR_ASSIGN_REF_CALLS.fetch_add(1, AtomicOrdering::Relaxed) + 1;
        if bitor_assign_sample_should_check(call_idx) {
            BITOR_ASSIGN_NOOP_SAMPLE_TOTAL.fetch_add(1, AtomicOrdering::Relaxed);
            if rhs.is_subset_of(self) {
                BITOR_ASSIGN_NOOP_SAMPLE_HIT.fetch_add(1, AtomicOrdering::Relaxed);
            }
        }
        if std::mem::discriminant(&*self) != std::mem::discriminant(rhs) {
            panic!("AbstractWeight operation requires both operands to be the same variant");
        }
        if rhs.is_empty() {
            BITOR_ASSIGN_RHS_EMPTY.fetch_add(1, AtomicOrdering::Relaxed);
            return;
        }
        if self.is_empty() {
            BITOR_ASSIGN_SELF_EMPTY.fetch_add(1, AtomicOrdering::Relaxed);
            *self = rhs.clone();
            return;
        }
        if self.is_all_fast() {
            BITOR_ASSIGN_SELF_ALL.fetch_add(1, AtomicOrdering::Relaxed);
            return;
        }
        if rhs.is_all_fast() {
            BITOR_ASSIGN_RHS_ALL.fetch_add(1, AtomicOrdering::Relaxed);
            *self = rhs.clone();
            return;
        }
        match (self, rhs) {
            (AbstractWeight::RangeSet(a), AbstractWeight::RangeSet(b)) => {
                BITOR_ASSIGN_UNION_TOTAL.fetch_add(1, AtomicOrdering::Relaxed);
                BITOR_ASSIGN_UNION_RANGESET.fetch_add(1, AtomicOrdering::Relaxed);
                WeightBackend::union_assign(a, b);
            }
            (AbstractWeight::Factorized(a), AbstractWeight::Factorized(b)) => {
                BITOR_ASSIGN_UNION_TOTAL.fetch_add(1, AtomicOrdering::Relaxed);
                BITOR_ASSIGN_UNION_FACTORIZED.fetch_add(1, AtomicOrdering::Relaxed);
                WeightBackend::union_assign(a, b);
            }
            (AbstractWeight::RangeMap(a), AbstractWeight::RangeMap(b)) => {
                BITOR_ASSIGN_UNION_TOTAL.fetch_add(1, AtomicOrdering::Relaxed);
                BITOR_ASSIGN_UNION_RANGEMAP.fetch_add(1, AtomicOrdering::Relaxed);
                WeightBackend::union_assign(a, b);
            }
            _ => panic!("AbstractWeight operation requires both operands to be the same variant"),
        }
    }
}

impl BitOrAssign<&&AbstractWeight> for AbstractWeight {
    fn bitor_assign(&mut self, rhs: &&AbstractWeight) {
        self.bitor_assign(*rhs);
    }
}

impl std::ops::Sub for AbstractWeight {
    type Output = AbstractWeight;

    fn sub(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (AbstractWeight::RangeSet(a), AbstractWeight::RangeSet(b)) => {
                AbstractWeight::RangeSet(WeightBackend::difference(&a, &b))
            }
            (AbstractWeight::Factorized(a), AbstractWeight::Factorized(b)) => {
                AbstractWeight::Factorized(WeightBackend::difference(&a, &b))
            }
            (AbstractWeight::RangeMap(a), AbstractWeight::RangeMap(b)) => {
                AbstractWeight::RangeMap(WeightBackend::difference(&a, &b))
            }
            _ => panic!("AbstractWeight operation requires both operands to be the same variant"),
        }
    }
}

impl std::ops::Sub for &AbstractWeight {
    type Output = AbstractWeight;

    fn sub(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (AbstractWeight::RangeSet(a), AbstractWeight::RangeSet(b)) => {
                AbstractWeight::RangeSet(WeightBackend::difference(a, b))
            }
            (AbstractWeight::Factorized(a), AbstractWeight::Factorized(b)) => {
                AbstractWeight::Factorized(WeightBackend::difference(a, b))
            }
            (AbstractWeight::RangeMap(a), AbstractWeight::RangeMap(b)) => {
                AbstractWeight::RangeMap(WeightBackend::difference(a, b))
            }
            _ => panic!("AbstractWeight operation requires both operands to be the same variant"),
        }
    }
}

impl std::ops::SubAssign<&AbstractWeight> for AbstractWeight {
    fn sub_assign(&mut self, rhs: &AbstractWeight) {
        match (self, rhs) {
            (AbstractWeight::RangeSet(a), AbstractWeight::RangeSet(b)) => {
                *a = WeightBackend::difference(a, b);
            }
            (AbstractWeight::Factorized(a), AbstractWeight::Factorized(b)) => {
                *a = WeightBackend::difference(a, b);
            }
            (AbstractWeight::RangeMap(a), AbstractWeight::RangeMap(b)) => {
                *a = WeightBackend::difference(a, b);
            }
            _ => panic!("AbstractWeight operation requires both operands to be the same variant"),
        }
    }
}

impl AbstractWeight {
    /// Compute the complement using global dimensions.
    pub fn complement(&self) -> Self {
        let max_llm_token = crate::datastructures::get_max_llm_token();
        let num_tsids = crate::datastructures::get_num_tsids();
        let domain_max = if num_tsids > 1 {
            max_llm_token
                .saturating_mul(num_tsids)
                .saturating_add(num_tsids.saturating_sub(1))
        } else {
            max_llm_token
        };
        match self {
            AbstractWeight::RangeSet(rsb) => {
                AbstractWeight::RangeSet(WeightBackend::complement(rsb, domain_max))
            }
            AbstractWeight::Factorized(fw) => {
                assert_eq!(
                    fw.num_tsids(),
                    normalize_num_tsids(num_tsids),
                    "FactorizedWeight dimensions mismatch in complement"
                );
                AbstractWeight::Factorized(WeightBackend::complement(fw, domain_max))
            }
            AbstractWeight::RangeMap(rm) => {
                assert_eq!(
                    rm.num_tsids(),
                    normalize_num_tsids(num_tsids),
                    "RangeMapWeight dimensions mismatch in complement"
                );
                AbstractWeight::RangeMap(WeightBackend::complement(rm, domain_max))
            }
        }
    }

    /// Compute a stable fingerprint for hashing/grouping weights.
    pub fn fingerprint(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }
    
    /// Iterate over positions up to and including max.
    /// 
    /// Note: This will expand factorized weights to RSB for iteration.
    /// For small test weights this is acceptable. For production hot paths,
    /// consider using contains() checks instead if possible.
    pub fn iter_up_to(&self, max: usize) -> impl Iterator<Item = usize> + '_ {
        match self {
            AbstractWeight::RangeSet(rsb) => {
                let clipped = rsb & &RangeSet::from(RangeSetBlaze::from_iter([0..=max]));
                clipped.into_iter()
            }
            AbstractWeight::Factorized(fw) => {
                fw.expand_to_rsb_bounded(max).into_iter()
            }
            AbstractWeight::RangeMap(rm) => {
                rm.expand_to_rsb_bounded(max).into_iter()
            }
        }
    }

    /// Iterate over positions up to and including max, allowing expansion.
    ///
    /// This should only be used in terminal-space weight operations where
    /// expansion is known to be safe and bounded.
    pub fn iter_up_to_allow_expansion(&self, max: usize) -> impl Iterator<Item = usize> + '_ {
        self.iter_up_to(max)
    }
    
    /// Compute the complement within the given dimensions.
    pub fn complement_with_dims(&self, dims: WeightDimensions) -> Self {
        if dims.domain_size() == 0 {
            return Self::empty();
        }
        match self {
            AbstractWeight::RangeSet(rsb) => {
                AbstractWeight::RangeSet(WeightBackend::complement(rsb, dims.max_position()))
            }
            AbstractWeight::Factorized(fw) => {
                assert_eq!(
                    fw.num_tsids(),
                    normalize_num_tsids(dims.num_tsids),
                    "FactorizedWeight dimensions mismatch in complement_with_dims"
                );
                AbstractWeight::Factorized(WeightBackend::complement(fw, dims.max_position()))
            }
            AbstractWeight::RangeMap(rm) => {
                assert_eq!(
                    rm.num_tsids(),
                    normalize_num_tsids(dims.num_tsids),
                    "RangeMapWeight dimensions mismatch in complement_with_dims"
                );
                AbstractWeight::RangeMap(WeightBackend::complement(rm, dims.max_position()))
            }
        }
    }

    /// Collect all unique interned RangeSet Arc pointers from this weight.
    /// 
    /// This is used for computing the true "interned" range count across
    /// multiple weights. Instead of counting by value equality, we count
    /// by Arc pointer identity - if two weights share the same interned
    /// RangeSet, it's only counted once.
    /// 
    /// Returns the set of Arc<RangeSetBlaze> raw pointers.
    pub fn collect_interned_rangesets(&self, seen: &mut std::collections::HashSet<usize>) {
        match self {
            AbstractWeight::RangeSet(rsb) => {
                // The inner Arc is the interned RangeSetBlaze
                seen.insert(Arc::as_ptr(&rsb.inner) as usize);
            }
            AbstractWeight::Factorized(fw) => {
                // Collect from all pairs
                for (tsid_set, token_set) in &fw.pairs {
                    seen.insert(Arc::as_ptr(&tsid_set.inner) as usize);
                    seen.insert(Arc::as_ptr(&token_set.inner) as usize);
                }
            }
            AbstractWeight::RangeMap(rm) => {
                // Collect from all map values
                for (_, tsid_set) in rm.map.range_values() {
                    seen.insert(Arc::as_ptr(&tsid_set.inner) as usize);
                }
            }
        }
    }

    /// Count ranges from unique interned RangeSets in this weight.
    /// 
    /// For multi-level interning, this sums up the ranges of each unique
    /// Arc<RangeSetBlaze> found within the weight structure.
    pub fn count_interned_ranges(&self) -> usize {
        match self {
            AbstractWeight::RangeSet(rsb) => rsb.ranges_len(),
            AbstractWeight::Factorized(fw) => {
                let mut seen: std::collections::HashSet<usize> = std::collections::HashSet::new();
                let mut total = 0;
                for (tsid_set, token_set) in &fw.pairs {
                    let ptr1 = Arc::as_ptr(&tsid_set.inner) as usize;
                    let ptr2 = Arc::as_ptr(&token_set.inner) as usize;
                    if seen.insert(ptr1) {
                        total += tsid_set.ranges_len();
                    }
                    if seen.insert(ptr2) {
                        total += token_set.ranges_len();
                    }
                }
                total
            }
            AbstractWeight::RangeMap(rm) => {
                let mut seen: std::collections::HashSet<usize> = std::collections::HashSet::new();
                let mut total = 0;
                for (_, tsid_set) in rm.map.range_values() {
                    let ptr = Arc::as_ptr(&tsid_set.inner) as usize;
                    if seen.insert(ptr) {
                        total += tsid_set.ranges_len();
                    }
                }
                total
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Conversion Traits
// ---------------------------------------------------------------------------

impl From<RangeSetBlaze<usize>> for AbstractWeight {
    fn from(rsb: RangeSetBlaze<usize>) -> Self {
        AbstractWeight::from_rsb(rsb)
    }
}

impl From<&RangeSetBlaze<usize>> for AbstractWeight {
    fn from(rsb: &RangeSetBlaze<usize>) -> Self {
        AbstractWeight::from_rsb(rsb.clone())
    }
}

impl From<crate::datastructures::hybrid_bitset::RangeSet> for AbstractWeight {
    fn from(rsb: crate::datastructures::hybrid_bitset::RangeSet) -> Self {
        AbstractWeight::from_rsb(std::sync::Arc::unwrap_or_clone(rsb.inner))
    }
}

impl From<&crate::datastructures::hybrid_bitset::RangeSet> for AbstractWeight {
    fn from(rsb: &crate::datastructures::hybrid_bitset::RangeSet) -> Self {
        AbstractWeight::from_rsb(std::sync::Arc::unwrap_or_clone(rsb.inner.clone()))
    }
}

impl From<FactorizedWeight> for AbstractWeight {
    fn from(weight: FactorizedWeight) -> Self {
        AbstractWeight::Factorized(intern_factorized(weight))
    }
}

impl From<&FactorizedWeight> for AbstractWeight {
    fn from(weight: &FactorizedWeight) -> Self {
        AbstractWeight::Factorized(intern_factorized(weight.clone()))
    }
}

impl From<RangeMapWeight> for AbstractWeight {
    fn from(weight: RangeMapWeight) -> Self {
        AbstractWeight::RangeMap(intern_rangemap(weight))
    }
}

impl From<&RangeMapWeight> for AbstractWeight {
    fn from(weight: &RangeMapWeight) -> Self {
        AbstractWeight::RangeMap(intern_rangemap(weight.clone()))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    type Backend = RangeSet;

    #[test]
    fn test_weight_dimensions_encoding() {
        let dims = WeightDimensions::new(100, 10);
        
        // Token 5, tsid 3 -> position 53
        assert_eq!(dims.encode(5, 3), 53);
        assert_eq!(dims.decode(53), (5, 3));
        
        // Token 0, tsid 0 -> position 0
        assert_eq!(dims.encode(0, 0), 0);
        assert_eq!(dims.decode(0), (0, 0));
        
        // Token 99, tsid 9 -> position 999
        assert_eq!(dims.encode(99, 9), 999);
        assert_eq!(dims.decode(999), (99, 9));
    }

    #[test]
    fn test_weight_dimensions_domain_size() {
        let dims = WeightDimensions::new(100, 10);
        assert_eq!(dims.domain_size(), 1000);
        assert_eq!(dims.max_position(), 999);
        assert!(dims.is_weight_heavy());
        
        let symbol_heavy = WeightDimensions::new(100, 1);
        assert_eq!(symbol_heavy.domain_size(), 100);
        assert!(!symbol_heavy.is_weight_heavy());
    }

    #[test]
    fn test_abstract_weight_basic() {
        let w = AbstractWeight::from_iter([1, 2, 3, 5, 6, 7]);
        assert_eq!(w.len(), 6);
        assert!(!w.is_empty());
        assert!(w.contains(1));
        assert!(w.contains(5));
        assert!(!w.contains(4));
    }

    #[test]
    fn test_abstract_weight_intersection() {
        let a = AbstractWeight::from_iter([1, 2, 3, 4, 5]);
        let b = AbstractWeight::from_iter([3, 4, 5, 6, 7]);
        let c = &a & &b;
        assert_eq!(c.len(), 3);
        assert!(c.contains(3));
        assert!(c.contains(4));
        assert!(c.contains(5));
    }

    #[test]
    fn test_abstract_weight_union() {
        let a = AbstractWeight::from_iter([1, 2, 3]);
        let b = AbstractWeight::from_iter([3, 4, 5]);
        let c = &a | &b;
        assert_eq!(c.len(), 5);
    }

    #[test]
    fn test_abstract_weight_all() {
        let dims = WeightDimensions::new(5, 3);
        let w = AbstractWeight::all_with_dims(dims);
        assert_eq!(w.len(), 15);
        assert!(w.contains(0));
        assert!(w.contains(14));
    }

    #[test]
    fn test_backend_ranges_and_len() {
        let backend = <Backend as WeightBackend>::from_ranges([1..=3, 7..=9]);
        assert_eq!(<Backend as WeightBackend>::ranges_len(&backend), 2);
        let ranges: Vec<(usize, usize)> = backend
            .ranges()
            .map(|r| (*r.start(), *r.end()))
            .collect();
        assert_eq!(ranges, vec![(1, 3), (7, 9)]);
        assert_eq!(<Backend as WeightBackend>::len(&backend), 6);
    }

    #[test]
    fn test_backend_set_ops_and_complement() {
        let a = <Backend as WeightBackend>::from_ranges([1..=4]);
        let b = <Backend as WeightBackend>::from_ranges([3..=6]);
        let intersect = <Backend as WeightBackend>::intersect(&a, &b);
        let union = <Backend as WeightBackend>::union(&a, &b);
        let diff = <Backend as WeightBackend>::difference(&a, &b);
        let comp = <Backend as WeightBackend>::complement(&a, 6);

        assert!(<Backend as WeightBackend>::contains(&intersect, 3));
        assert!(!<Backend as WeightBackend>::contains(&intersect, 2));
        assert!(<Backend as WeightBackend>::contains(&union, 6));
        assert!(<Backend as WeightBackend>::contains(&diff, 1));
        assert!(!<Backend as WeightBackend>::contains(&diff, 5));
        assert!(!<Backend as WeightBackend>::contains(&comp, 2));
        assert!(<Backend as WeightBackend>::contains(&comp, 6));
    }

    #[test]
    fn test_backend_min_max() {
        let backend = <Backend as WeightBackend>::from_ranges([2..=4, 8..=10]);
        assert_eq!(<Backend as WeightBackend>::min_item(&backend), Some(2));
        assert_eq!(<Backend as WeightBackend>::max_item(&backend), Some(10));
    }

    #[test]
    fn test_factorized_expand_roundtrip() {
        // Allow expansion for this test
        let prev = override_expansion_allowed(true);
        let num_tsids = 3;
        let rsb = RangeSetBlaze::from_iter([0..=2, 4..=5, 8..=8, 10..=12]);
        let fw = FactorizedWeight::from_rsb_with_num_tsids(&rsb, num_tsids);
        assert_eq!(fw.expand_to_rsb(), rsb);
        restore_expansion_allowed(prev);
    }

    #[test]
    fn test_factorized_set_ops_match_rsb() {
        // Allow expansion for this test
        let prev = override_expansion_allowed(true);
        let num_tsids = 4;
        let a_rsb = RangeSetBlaze::from_iter([0..=7, 10..=12]);
        let b_rsb = RangeSetBlaze::from_iter([5..=9, 12..=15]);
        let a = FactorizedWeight::from_rsb_with_num_tsids(&a_rsb, num_tsids);
        let b = FactorizedWeight::from_rsb_with_num_tsids(&b_rsb, num_tsids);

        let inter = <FactorizedWeight as WeightBackend>::intersect(&a, &b);
        let union = <FactorizedWeight as WeightBackend>::union(&a, &b);

        assert_eq!(inter.expand_to_rsb(), &a_rsb & &b_rsb);
        assert_eq!(union.expand_to_rsb(), &a_rsb | &b_rsb);
        restore_expansion_allowed(prev);
    }
}
