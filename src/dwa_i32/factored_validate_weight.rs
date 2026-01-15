//! FactoredValidateWeight: stores both FactoredWeight and RangeSet for validation.
//!
//! This backend is meant for debugging and correctness validation. It computes
//! operations in both representations and asserts that they match within the
//! valid N×M domain.

use range_set_blaze::RangeSetBlaze;

use crate::dwa_i32::factored_weight::FactoredWeight;
use crate::dwa_i32::rangeset::RangeSet;
use crate::dwa_i32::get_weight_dimensions;

#[derive(Clone)]
pub struct FactoredValidateWeight {
    factored: FactoredWeight,
    rangeset: RangeSet,
}

impl std::fmt::Debug for FactoredValidateWeight {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FactoredValidateWeight")
            .field("factored_terms", &self.factored.num_terms())
            .field("rangeset_len", &self.rangeset.len())
            .finish()
    }
}

impl FactoredValidateWeight {
    pub fn new(factored: FactoredWeight, rangeset: RangeSet) -> Self {
        let weight = Self { factored, rangeset };
        weight.validate();
        weight
    }

    pub fn factored(&self) -> &FactoredWeight {
        &self.factored
    }

    pub fn rangeset(&self) -> &RangeSet {
        &self.rangeset
    }

    pub fn validate(&self) {
        let dims = get_weight_dimensions();
        let domain_max = dims.num_tokens.saturating_mul(dims.num_tsids);
        let max = domain_max.saturating_sub(1);

        let start = std::time::Instant::now();

        let mut clipped = self.rangeset.clone();
        if domain_max > 0 {
            clipped.clip_max(max);
        }

        let expanded: RangeSetBlaze<usize> = self.factored.expand_impl();
        let rsb: RangeSetBlaze<usize> = clipped.rsb.clone();

        let elapsed = start.elapsed();
        if elapsed.as_millis() >= 5 {
            crate::debug!(5, "FactoredValidate validate: terms={} rs_ranges={} expanded_ranges={} len={} elapsed={:?}",
                self.factored.num_terms(),
                rsb.ranges_len(),
                expanded.ranges().count(),
                rsb.len(),
                elapsed
            );
        }

        if expanded != rsb {
            let only_factored = &expanded - &rsb;
            let only_rangeset = &rsb - &expanded;
            panic!(
                "FactoredValidate mismatch: factored_ranges={} rs_ranges={} only_factored={} only_rangeset={} max_pos={}",
                expanded.ranges().count(),
                rsb.ranges().count(),
                only_factored.ranges().count(),
                only_rangeset.ranges().count(),
                max
            );
        }
    }
}
