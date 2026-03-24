use std::ops::RangeInclusive;

use super::dwa::DWA;
use super::nwa::NWA;
use crate::ds::weight::Weight;

pub(crate) fn add_dwa_states(dwa: &mut DWA, count: usize) {
    for _ in 0..count {
        dwa.add_state();
    }
}

pub(crate) fn add_nwa_states(nwa: &mut NWA, count: usize) {
    for _ in 0..count {
        nwa.add_state();
    }
}

pub(crate) fn weight_from_item(item: u32) -> Weight {
    Weight::from_compact_ranges(vec![(item..=item, vec![0..=0])])
}

pub(crate) fn weight_from_iter<I: IntoIterator<Item = u32>>(items: I) -> Weight {
    let mut sorted: Vec<u32> = items.into_iter().collect();
    if sorted.is_empty() {
        return Weight::empty();
    }

    sorted.sort_unstable();
    sorted.dedup();

    let mut ranges = Vec::new();
    let mut start = sorted[0];
    let mut end = sorted[0];
    for &item in &sorted[1..] {
        if item == end + 1 {
            end = item;
        } else {
            ranges.push((start..=end, vec![0..=0]));
            start = item;
            end = item;
        }
    }
    ranges.push((start..=end, vec![0..=0]));

    Weight::from_compact_ranges(ranges)
}

pub(crate) fn weight_from_range(start: u32, end: u32) -> Weight {
    weight_from_ranges([start..=end])
}

pub(crate) fn weight_from_ranges<I: IntoIterator<Item = RangeInclusive<u32>>>(ranges: I) -> Weight {
    let entries: Vec<_> = ranges
        .into_iter()
        .map(|range| (range, vec![0..=0]))
        .collect();
    if entries.is_empty() {
        return Weight::empty();
    }

    Weight::from_compact_ranges(entries)
}

pub(crate) fn weight_contains(weight: &Weight, item: u32) -> bool {
    !weight.intersection(&weight_from_item(item)).is_empty()
}

pub(crate) fn assert_weights_eq(left: &Weight, right: &Weight, message: &str) {
    let left_only = left.difference(right);
    let right_only = right.difference(left);
    assert!(
        left_only.is_empty() && right_only.is_empty(),
        "{}\nleft: {}\nright: {}\nleft\\right: {}\nright\\left: {}",
        message,
        left,
        right,
        left_only,
        right_only
    );
}