//! RangeMap-based weight backend.
//!
//! Stores transition weights as sorted ranges. Uses TSID-outer layout
//! (keyed by token-set ID, then state) for cache-friendly mask computation.

use serde::{Deserialize, Serialize};

/// A token-set ID. Groups of tokens that behave identically through a
/// DWA state transition share the same TSID.
pub type Tsid = u32;

/// Represents a mapping from non-overlapping ranges of token-set IDs to values.
///
/// Stored as sorted `(start, end, value)` triples representing half-open ranges `[start, end)`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RangeMap<V> {
    /// Sorted entries: `(start, end, value)` where range is `[start, end)`.
    entries: Vec<(u32, u32, V)>,
}

impl<V: Clone + Eq> RangeMap<V> {
    /// Create an empty range map.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Create from pre-sorted entries.
    pub fn from_sorted(entries: Vec<(u32, u32, V)>) -> Self {
        Self { entries }
    }

    /// Number of range entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Look up the value for a given key using binary search.
    pub fn get(&self, key: u32) -> Option<&V> {
        let idx = self
            .entries
            .binary_search_by(|&(start, end, _)| {
                if key < start {
                    std::cmp::Ordering::Greater
                } else if key >= end {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Equal
                }
            })
            .ok()?;
        Some(&self.entries[idx].2)
    }

    /// Iterate over all entries as `(start, end, &value)`.
    pub fn iter(&self) -> impl Iterator<Item = (u32, u32, &V)> {
        self.entries.iter().map(|&(s, e, ref v)| (s, e, v))
    }

    /// Access entries as a slice.
    pub fn entries(&self) -> &[(u32, u32, V)] {
        &self.entries
    }
}

impl<V: Clone + Eq> Default for RangeMap<V> {
    fn default() -> Self {
        Self::new()
    }
}

/// Weight layout using TSID-outer organization.
///
/// For each (tsid, state) pair, stores the resulting DWA transition (target state + weight).
/// The outer dimension is TSID so that computing a mask for a single token set
/// requires a contiguous memory scan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightTable {
    /// Number of DWA states.
    pub num_states: u32,
    /// Number of token-set IDs.
    pub num_tsids: u32,
    /// Flat table: `data[tsid * num_states + state] = (target_state, weight)`.
    /// `target_state == u32::MAX` means dead/no transition.
    pub data: Vec<(u32, i32)>,
}

impl WeightTable {
    /// Create a new weight table with all dead transitions.
    pub fn new(num_states: u32, num_tsids: u32) -> Self {
        let size = num_states as usize * num_tsids as usize;
        Self {
            num_states,
            num_tsids,
            data: vec![(u32::MAX, 0); size],
        }
    }

    /// Get the transition for `(tsid, state)`.
    #[inline]
    pub fn get(&self, tsid: u32, state: u32) -> (u32, i32) {
        self.data[tsid as usize * self.num_states as usize + state as usize]
    }

    /// Set the transition for `(tsid, state)`.
    #[inline]
    pub fn set(&mut self, tsid: u32, state: u32, target: u32, weight: i32) {
        self.data[tsid as usize * self.num_states as usize + state as usize] = (target, weight);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_range_map_lookup() {
        let rm = RangeMap::from_sorted(vec![
            (0, 10, "a"),
            (10, 20, "b"),
            (30, 40, "c"),
        ]);
        assert_eq!(rm.get(0), Some(&"a"));
        assert_eq!(rm.get(9), Some(&"a"));
        assert_eq!(rm.get(10), Some(&"b"));
        assert_eq!(rm.get(25), None);
        assert_eq!(rm.get(35), Some(&"c"));
    }

    #[test]
    fn test_weight_table() {
        let mut wt = WeightTable::new(3, 2);
        wt.set(0, 1, 2, 5);
        assert_eq!(wt.get(0, 1), (2, 5));
        assert_eq!(wt.get(1, 0), (u32::MAX, 0)); // untouched = dead
    }
}
