//! Numeric thresholds for Mask fast paths.

pub(super) const DELTA_SEED_MIN_SAVINGS: u64 = 2048;
pub(super) const MASK_SINGLE_PATH_DIRECT_MAX_DEPTH: u32 = 64;
pub(super) const MASK_SINGLE_PATH_DIRECT_MAX_TOTAL_PATHS: usize = 8;
