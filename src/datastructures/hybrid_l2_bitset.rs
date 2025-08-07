use crate::datastructures::cache::{self, Acc};
use crate::datastructures::hybrid_bitset::HybridBitset;
use range_set_blaze::prelude::*;
use range_set_blaze::RangeMapBlaze;
use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::fmt::{Debug, Formatter};
use std::hash::{Hash, Hasher};
use std::iter::FromIterator;
use std::ops::{
    BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, RangeInclusive, Sub, SubAssign,
};
use std::sync::Arc;

#[derive(Clone, Eq)]
pub struct HybridL2Bitset {
    pub(crate) inner: Acc<RangeMapBlaze<usize, HybridBitset>>,
}

impl Debug for HybridL2Bitset {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let f_alternate = f.alternate();
        let mut ds = f.debug_struct("HybridL2Bitset");

        const MAX_RANGES_TO_SHOW: usize = 5;
        let total_ranges = self.inner.ranges_len();

        if f_alternate || total_ranges <= MAX_RANGES_TO_SHOW {
            ds.field("inner", &self.inner);
        } else {
            let ranges_to_show: Vec<_> = self.inner.range_values().take(MAX_RANGES_TO_SHOW).collect();
            ds.field("inner_preview", &ranges_to_show);
            ds.field(
                "...",
                &format_args!("and {} more ranges", total_ranges - MAX_RANGES_TO_SHOW),
            );
        }

        ds.finish()
    }
}

impl PartialOrd for HybridL2Bitset {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HybridL2Bitset {
    fn cmp(&self, other: &Self) -> Ordering {
        if Arc::ptr_eq(&self.inner, &other.inner) {
            return Ordering::Equal;
        }
        // Manual comparison because RangeMapBlaze doesn't impl Ord directly on its values.
        let mut self_iter = self.inner.range_values();
        let mut other_iter = other.inner.range_values();

        loop {
            match (self_iter.next(), other_iter.next()) {
                (Some((r1, v1)), Some((r2, v2))) => {
                    let range_cmp = r1.start().cmp(r2.start()).then_with(|| r1.end().cmp(r2.end()));
                    if range_cmp != Ordering::Equal {
                        return range_cmp;
                    }
                    let value_cmp = v1.cmp(v2);
                    if value_cmp != Ordering::Equal {
                        return value_cmp;
                    }
                }
                (Some(_), None) => return Ordering::Greater,
                (None, Some(_)) => return Ordering::Less,
                (None, None) => return Ordering::Equal,
            }
        }
    }
}

impl PartialEq for HybridL2Bitset {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner) || *self.inner == *other.inner
    }
}

impl Hash for HybridL2Bitset {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner.hash(state);
    }
}

impl HybridL2Bitset {
    pub fn new() -> Self {
        HybridL2Bitset {
            inner: cache::intern_l2(RangeMapBlaze::new()),
        }
    }

    pub fn all() -> Self {
        HybridL2Bitset {
            inner: cache::intern_l2(RangeMapBlaze::from_iter(std::iter::once((
                0..=usize::MAX,
                HybridBitset::max_ones(),
            )))),
        }
    }

    pub fn is_simple(&self) -> bool {
        self.inner.ranges_len() < cache::SIMPLE_L2_BITSET_THRESHOLD
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn len(&self) -> usize {
        self.inner.iter().map(|(_, bitset)| bitset.len()).sum()
    }

    pub fn clear(&mut self) {
        self.inner = cache::intern_l2(RangeMapBlaze::new());
    }

    pub fn insert(&mut self, l1_index: usize, l2_index: usize) {
        let mut new_inner = (*self.inner).clone();
        let mut bitset = new_inner
            .remove(l1_index)
            .unwrap_or_else(HybridBitset::zeros);
        bitset.insert(l2_index); // This modifies bitset, replacing its inner Arc
        if !bitset.is_empty() {
            new_inner.insert(l1_index, bitset);
        }
        self.inner = cache::intern_l2(new_inner);
    }

    pub fn insert_l2_bitset(&mut self, l1_index: usize, bitset: HybridBitset) {
        let mut new_inner = (*self.inner).clone();
        if !bitset.is_empty() {
            new_inner.insert(l1_index, bitset);
        } else {
            new_inner.remove(l1_index);
        }
        self.inner = cache::intern_l2(new_inner);
    }

    pub fn remove(&mut self, l1_index: usize, l2_index: usize) -> bool {
        if let Some(original_l2) = self.inner.get(l1_index) {
            if !original_l2.contains(l2_index) {
                return false;
            }
            let mut new_inner = (*self.inner).clone();
            let mut bitset = new_inner.remove(l1_index).unwrap();
            let was_present = bitset.remove(l2_index);
            if !bitset.is_empty() {
                new_inner.insert(l1_index, bitset);
            }
            self.inner = cache::intern_l2(new_inner);
            was_present
        } else {
            false
        }
    }

    pub fn remove_l1(&mut self, l1_index: usize) -> Option<HybridBitset> {
        let mut new_inner = (*self.inner).clone();
        let result = new_inner.remove(l1_index);
        self.inner = cache::intern_l2(new_inner);
        result
    }

    pub fn contains(&self, l1_index: usize, l2_index: usize) -> bool {
        self.inner
            .get(l1_index)
            .map_or(false, |bitset| bitset.contains(l2_index))
    }

    pub fn get_l2_bitset(&self, l1_index: usize) -> Option<&HybridBitset> {
        self.inner.get(l1_index)
    }

    pub fn iter(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.inner
            .iter()
            .flat_map(|(l1_index, bitset)| bitset.iter().map(move |l2_index| (l1_index, l2_index)))
    }

    pub fn iter_l1_bitsets(&self) -> impl Iterator<Item = (usize, &HybridBitset)> {
        self.inner.iter()
    }

    pub fn range_values(&self) -> impl Iterator<Item = (RangeInclusive<usize>, &HybridBitset)> {
        self.inner.range_values()
    }

    pub fn complement(&self) -> Self {
        let complemented_values = self
            .inner
            .range_values()
            .map(|(range, bitset)| (range, bitset.inverted()));

        HybridL2Bitset {
            inner: cache::intern_l2(complemented_values.collect()),
        }
    }

    pub fn intersection_with(&self, other: &Self, default: Option<HybridBitset>) -> Self {
        self.zip_op(other, default, |a, b| a & b)
    }

    pub fn union_with(&self, other: &Self, default: Option<HybridBitset>) -> Self {
        self.zip_op(other, default, |a, b| a | b)
    }

    pub fn symmetric_difference_with(&self, other: &Self, default: Option<HybridBitset>) -> Self {
        self.zip_op(other, default, |a, b| a ^ b)
    }

    fn zip_op<F>(&self, other: &Self, default: Option<HybridBitset>, op: F) -> Self
    where
        F: Fn(&HybridBitset, &HybridBitset) -> HybridBitset,
    {
        let mut points = BTreeSet::new();
        if !self.inner.is_empty() || !other.inner.is_empty() || default.is_some() {
            points.insert(0);
        }

        for (range, _) in self.inner.range_values() {
            points.insert(*range.start());
            if *range.end() < usize::MAX {
                points.insert(range.end() + 1);
            }
        }
        for (range, _) in other.inner.range_values() {
            points.insert(*range.start());
            if *range.end() < usize::MAX {
                points.insert(range.end() + 1);
            }
        }

        let mut new_ranges = Vec::new();

        let mut process_interval = |start, end| {
            if start > end {
                return;
            }
            let mid = start;

            let self_bs_opt = self.inner.get(mid);
            let other_bs_opt = other.inner.get(mid);

            let result_bs = match (self_bs_opt, other_bs_opt) {
                (Some(s), Some(o)) => op(s, o),
                (Some(s), None) => {
                    if let Some(ref d) = default {
                        op(s, d)
                    } else {
                        return;
                    }
                }
                (None, Some(o)) => {
                    if let Some(ref d) = default {
                        op(d, o)
                    } else {
                        return;
                    }
                }
                (None, None) => {
                    if let Some(ref d) = default {
                        op(d, d)
                    } else {
                        return;
                    }
                }
            };

            if !result_bs.is_empty() {
                new_ranges.push((start..=end, result_bs));
            }
        };

        let points_vec: Vec<_> = points.into_iter().collect();
        for window in points_vec.windows(2) {
            process_interval(window[0], window[1] - 1);
        }

        if let Some(&last_point) = points_vec.last() {
            if last_point <= usize::MAX {
                process_interval(last_point, usize::MAX);
            }
        }

        HybridL2Bitset {
            inner: cache::intern_l2(RangeMapBlaze::from_iter(new_ranges)),
        }
    }
}

impl Default for HybridL2Bitset {
    fn default() -> Self {
        Self::new()
    }
}

// --- Bitwise Operations ---

impl BitAnd for &HybridL2Bitset {
    type Output = HybridL2Bitset;

    fn bitand(self, rhs: Self) -> Self::Output {
        if Arc::ptr_eq(&self.inner, &rhs.inner) {
            return self.clone();
        }
        if self.is_simple() || rhs.is_simple() {
            return self.intersection_with(rhs, None);
        }
        if let Some(cached) = cache::get_l2_op_cache(cache::BinOp::And, &self.inner, &rhs.inner) {
            return HybridL2Bitset { inner: cached };
        }
        if let Some(cached) = cache::get_l2_op_cache(cache::BinOp::And, &rhs.inner, &self.inner) {
            return HybridL2Bitset { inner: cached };
        }
        let result = self.intersection_with(rhs, None);
        cache::put_l2_op_cache(
            cache::BinOp::And,
            self.inner.clone(),
            rhs.inner.clone(),
            result.inner.clone(),
        );
        result
    }
}

impl BitOr for &HybridL2Bitset {
    type Output = HybridL2Bitset;

    fn bitor(self, rhs: Self) -> Self::Output {
        if Arc::ptr_eq(&self.inner, &rhs.inner) {
            return self.clone();
        }
        if self.is_simple() || rhs.is_simple() {
            return self.union_with(rhs, Some(HybridBitset::zeros()));
        }
        if let Some(cached) = cache::get_l2_op_cache(cache::BinOp::Or, &self.inner, &rhs.inner) {
            return HybridL2Bitset { inner: cached };
        }
        if let Some(cached) = cache::get_l2_op_cache(cache::BinOp::Or, &rhs.inner, &self.inner) {
            return HybridL2Bitset { inner: cached };
        }
        let result = self.union_with(rhs, Some(HybridBitset::zeros()));
        cache::put_l2_op_cache(
            cache::BinOp::Or,
            self.inner.clone(),
            rhs.inner.clone(),
            result.inner.clone(),
        );
        result
    }
}

impl BitXor for &HybridL2Bitset {
    type Output = HybridL2Bitset;

    fn bitxor(self, rhs: Self) -> Self::Output {
        if Arc::ptr_eq(&self.inner, &rhs.inner) {
            return Self::new();
        }
        if self.is_simple() || rhs.is_simple() {
            return self.symmetric_difference_with(rhs, Some(HybridBitset::zeros()));
        }
        if let Some(cached) = cache::get_l2_op_cache(cache::BinOp::Xor, &self.inner, &rhs.inner) {
            return HybridL2Bitset { inner: cached };
        }
        if let Some(cached) = cache::get_l2_op_cache(cache::BinOp::Xor, &rhs.inner, &self.inner) {
            return HybridL2Bitset { inner: cached };
        }
        let result = self.symmetric_difference_with(rhs, Some(HybridBitset::zeros()));
        cache::put_l2_op_cache(
            cache::BinOp::Xor,
            self.inner.clone(),
            rhs.inner.clone(),
            result.inner.clone(),
        );
        result
    }
}

impl Sub for &HybridL2Bitset {
    type Output = HybridL2Bitset;
    fn sub(self, rhs: Self) -> Self::Output {
        if Arc::ptr_eq(&self.inner, &rhs.inner) {
            return Self::new();
        }
        if self.is_simple() || rhs.is_simple() {
            return self.zip_op(rhs, Some(HybridBitset::zeros()), |a, b| a - b);
        }
        if let Some(cached) = cache::get_l2_op_cache(cache::BinOp::Sub, &self.inner, &rhs.inner) {
            return HybridL2Bitset { inner: cached };
        }
        let result = self.zip_op(rhs, Some(HybridBitset::zeros()), |a, b| a - b);
        cache::put_l2_op_cache(
            cache::BinOp::Sub,
            self.inner.clone(),
            rhs.inner.clone(),
            result.inner.clone(),
        );
        result
    }
}

// --- In-place Bitwise Operations ---
impl BitAndAssign for HybridL2Bitset {
    fn bitand_assign(&mut self, rhs: Self) {
        *self = &*self & &rhs;
    }
}
impl BitAndAssign<&HybridL2Bitset> for HybridL2Bitset {
    fn bitand_assign(&mut self, rhs: &HybridL2Bitset) {
        *self = &*self & rhs;
    }
}
impl BitOrAssign for HybridL2Bitset {
    fn bitor_assign(&mut self, rhs: Self) {
        *self = &*self | &rhs;
    }
}
impl BitOrAssign<&HybridL2Bitset> for HybridL2Bitset {
    fn bitor_assign(&mut self, rhs: &HybridL2Bitset) {
        *self = &*self | rhs;
    }
}
impl BitXorAssign for HybridL2Bitset {
    fn bitxor_assign(&mut self, rhs: Self) {
        *self = &*self ^ &rhs;
    }
}
impl BitXorAssign<&HybridL2Bitset> for HybridL2Bitset {
    fn bitxor_assign(&mut self, rhs: &HybridL2Bitset) {
        *self = &*self ^ rhs;
    }
}
impl SubAssign for HybridL2Bitset {
    fn sub_assign(&mut self, rhs: Self) {
        *self = &*self - &rhs;
    }
}
impl SubAssign<&HybridL2Bitset> for HybridL2Bitset {
    fn sub_assign(&mut self, rhs: &HybridL2Bitset) {
        *self = &*self - rhs;
    }
}

// --- Operations on owned values ---
impl BitAnd<HybridL2Bitset> for HybridL2Bitset {
    type Output = HybridL2Bitset;
    fn bitand(self, rhs: HybridL2Bitset) -> Self::Output {
        &self & &rhs
    }
}
impl BitOr<HybridL2Bitset> for HybridL2Bitset {
    type Output = HybridL2Bitset;
    fn bitor(self, rhs: HybridL2Bitset) -> Self::Output {
        &self | &rhs
    }
}
impl BitXor<HybridL2Bitset> for HybridL2Bitset {
    type Output = HybridL2Bitset;
    fn bitxor(self, rhs: HybridL2Bitset) -> Self::Output {
        &self ^ &rhs
    }
}
impl Sub<HybridL2Bitset> for HybridL2Bitset {
    type Output = HybridL2Bitset;
    fn sub(self, rhs: HybridL2Bitset) -> Self::Output {
        &self - &rhs
    }
}

impl<'a> BitAnd<&'a HybridL2Bitset> for HybridL2Bitset {
    type Output = HybridL2Bitset;
    fn bitand(self, rhs: &'a HybridL2Bitset) -> Self::Output {
        &self & rhs
    }
}
impl<'a> BitOr<&'a HybridL2Bitset> for HybridL2Bitset {
    type Output = HybridL2Bitset;
    fn bitor(self, rhs: &'a HybridL2Bitset) -> Self::Output {
        &self | rhs
    }
}
impl<'a> BitXor<&'a HybridL2Bitset> for HybridL2Bitset {
    type Output = HybridL2Bitset;
    fn bitxor(self, rhs: &'a HybridL2Bitset) -> Self::Output {
        &self ^ rhs
    }
}
impl<'a> Sub<&'a HybridL2Bitset> for HybridL2Bitset {
    type Output = HybridL2Bitset;
    fn sub(self, rhs: &'a HybridL2Bitset) -> Self::Output {
        &self - rhs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn test_new_and_is_empty() {
        let set = HybridL2Bitset::new();
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);
    }

    #[test]
    fn test_insert_and_contains() {
        let mut set = HybridL2Bitset::new();
        assert!(!set.contains(10, 100));

        set.insert(10, 100);
        assert!(set.contains(10, 100));
        assert!(!set.contains(10, 101));
        assert!(!set.contains(11, 100));
        assert!(!set.is_empty());

        set.insert(10, 101);
        assert!(set.contains(10, 100));
        assert!(set.contains(10, 101));

        set.insert(11, 100);
        assert!(set.contains(11, 100));
    }

    #[test]
    fn test_remove() {
        let mut set = HybridL2Bitset::new();
        set.insert(10, 100);
        set.insert(10, 101);
        set.insert(20, 200);

        assert!(set.remove(10, 100));
        assert!(!set.contains(10, 100));
        assert!(set.contains(10, 101));
        assert!(set.contains(20, 200));

        assert!(!set.remove(10, 99)); // Was not present

        // Removing the last element of a row should remove the row
        assert!(set.get_l2_bitset(10).is_some());
        assert!(set.remove(10, 101));
        assert!(!set.contains(10, 101));
        assert!(set.get_l2_bitset(10).is_none());

        assert!(!set.remove(30, 300)); // Row not present
    }

    #[test]
    fn test_len_and_clear() {
        let mut set = HybridL2Bitset::new();
        set.insert(1, 10);
        set.insert(1, 20);
        set.insert(1, 30);
        set.insert(100, 10);
        set.insert(100, 20);

        assert_eq!(set.len(), 5);

        set.remove(1, 20);
        assert_eq!(set.len(), 4);

        set.clear();
        assert_eq!(set.len(), 0);
        assert!(set.is_empty());
    }

    #[test]
    fn test_iter() {
        let mut set = HybridL2Bitset::new();
        set.insert(10, 100);
        set.insert(2, 50);
        set.insert(10, 101);
        set.insert(5, 80);

        let expected: BTreeSet<(usize, usize)> = vec![(2, 50), (5, 80), (10, 100), (10, 101)]
            .into_iter()
            .collect();

        let collected: BTreeSet<(usize, usize)> = set.iter().collect();
        assert_eq!(collected, expected);

        let empty_set = HybridL2Bitset::new();
        assert_eq!(empty_set.iter().count(), 0);
    }

    #[test]
    fn test_bitwise_ops() {
        let mut set1 = HybridL2Bitset::new();
        set1.insert(10, 100);
        set1.insert(10, 101);
        set1.insert(20, 200);

        let mut set2 = HybridL2Bitset::new();
        set2.insert(11, 101);
        set2.insert(11, 102);
        set2.insert(20, 200);
        set2.insert(30, 300);

        // Intersection
        let intersection = &set1 & &set2;
        let mut expected_intersection_set = HybridL2Bitset::new();
        expected_intersection_set.insert(20, 200);
        assert_eq!(intersection, expected_intersection_set);

        // Union
        let union = &set1 | &set2;
        let mut expected_union_set = HybridL2Bitset::new();
        expected_union_set.insert(10, 100);
        expected_union_set.insert(10, 101);
        expected_union_set.insert(11, 101);
        expected_union_set.insert(11, 102);
        expected_union_set.insert(20, 200);
        expected_union_set.insert(30, 300);
        assert_eq!(union, expected_union_set);

        // Symmetric Difference (XOR)
        let xor = &set1 ^ &set2;
        let mut expected_xor_set = HybridL2Bitset::new();
        expected_xor_set.insert(10, 100);
        expected_xor_set.insert(10, 101);
        expected_xor_set.insert(11, 101);
        expected_xor_set.insert(11, 102);
        expected_xor_set.insert(30, 300);
        assert_eq!(xor, expected_xor_set);
    }

    #[test]
    fn test_bitwise_assign_ops() {
        let mut set1_orig = HybridL2Bitset::new();
        set1_orig.insert(10, 100);
        set1_orig.insert(20, 200);

        let mut set2 = HybridL2Bitset::new();
        set2.insert(20, 200);
        set2.insert(30, 300);

        let mut set1_and = set1_orig.clone();
        set1_and &= &set2;
        let mut expected_and = HybridL2Bitset::new();
        expected_and.insert(20, 200);
        assert_eq!(set1_and, expected_and);

        let mut set1_or = set1_orig.clone();
        set1_or |= &set2;
        let mut expected_or = HybridL2Bitset::new();
        expected_or.insert(10, 100);
        expected_or.insert(20, 200);
        expected_or.insert(30, 300);
        assert_eq!(set1_or, expected_or);

        let mut set1_xor = set1_orig.clone();
        set1_xor ^= &set2;
        let mut expected_xor = HybridL2Bitset::new();
        expected_xor.insert(10, 100);
        expected_xor.insert(30, 300);
        assert_eq!(set1_xor, expected_xor);
    }

    #[test]
    fn test_get_l2_bitset() {
        let mut set = HybridL2Bitset::new();
        set.insert(5, 50);
        set.insert(5, 51);

        let l2_set = set.get_l2_bitset(5).unwrap();
        assert_eq!(l2_set.len(), 2);
        assert!(l2_set.contains(50));
        assert!(l2_set.contains(51));

        assert!(set.get_l2_bitset(99).is_none());
    }

    #[test]
    fn test_with_ops() {
        let mut set1 = HybridL2Bitset::new();
        set1.insert(10, 100); // in set1 only
        set1.insert(20, 200); // in both

        let mut set2 = HybridL2Bitset::new();
        set2.insert(20, 201); // in both
        set2.insert(30, 300); // in set2 only

        // intersection_with
        // with None default (same as bitand)
        let inter_none = set1.intersection_with(&set2, None);
        let expected_inter = HybridL2Bitset::new();
        // key 20 is in both. intersection of {200} and {201} is empty.
        assert!(inter_none.is_empty());
        assert_eq!(inter_none, &set1 & &set2);

        // with non-empty default
        let default_bs = HybridBitset::from_iter(vec![100, 200, 300]);
        let inter_default = set1.intersection_with(&set2, Some(default_bs.clone()));
        // key 10: {100} & {100,200,300} -> {100}
        // key 20: {200} & {201} -> {}
        // key 30: {100,200,300} & {300} -> {300}
        let mut expected_inter_default = HybridL2Bitset::new();
        expected_inter_default.insert(10, 100);
        expected_inter_default.insert(30, 300);
        assert_eq!(inter_default, expected_inter_default);

        // union_with
        // with None default (only common keys)
        let union_none = set1.union_with(&set2, None);
        // key 10: excluded
        // key 20: {200} | {201} -> {200, 201}
        // key 30: excluded
        let mut expected_union_none = HybridL2Bitset::new();
        expected_union_none.insert(20, 200);
        expected_union_none.insert(20, 201);
        assert_eq!(union_none, expected_union_none);

        // with zeros default (same as bitor)
        let union_zeros = set1.union_with(&set2, Some(HybridBitset::zeros()));
        let mut expected_union_zeros = HybridL2Bitset::new();
        expected_union_zeros.insert(10, 100);
        expected_union_zeros.insert(20, 200);
        expected_union_zeros.insert(20, 201);
        expected_union_zeros.insert(30, 300);
        assert_eq!(union_zeros, expected_union_zeros);
        assert_eq!(union_zeros, &set1 | &set2);
    }
}
