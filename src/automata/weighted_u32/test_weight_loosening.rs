//! Regression tests for weight loosening.
//!
//! Source had 3 tests; 2 depend on `loosen_weights_for_minimize()` (absent in glrmask) and are
//! skipped. 1 test is retained here.
//!
//! Skipped tests:
//!   - test_weight_loosening_preserves_semantics  (needs loosen_weights_for_minimize)
//!   - test_weight_loosening_skips_cyclic          (needs loosen_weights_for_minimize + is_cyclic)

use super::dwa::DWA;
use super::minimize;
use super::nwa::Label;
use crate::ds::weight::Weight;

// ── helpers ──────────────────────────────────────────────────────────────────

/// Build a Weight that contains a single TSID (token-set-id) item, encoded
/// as a 2-D range `(N..=N, 0..=0)`.
fn weight_from_item(n: u32) -> Weight {
    Weight::from_compact_ranges([(n..=n, [0u32..=0])])
}

/// Build a Weight that spans a contiguous range of TSIDs `lo..=hi`, encoded
/// as `(lo..=hi, 0..=0)`.
fn weight_from_range(lo: u32, hi: u32) -> Weight {
    Weight::from_compact_ranges([(lo..=hi, [0u32..=0])])
}

// ── regression tests ─────────────────────────────────────────────────────────

/// Adapted from `test_weight_loosening_loosens_unreachable`.
///
/// DWA structure:
/// ```text
///   State 0 (start) --1--> State 1 (final, weight=[5..=10])
///                    w=[0]
/// ```
///
/// Only TSID 0 can reach state 1, but state 1 only accepts TSIDs 5‥10.
/// Since 0 ∉ [5,10], no word is accepted.
/// After minimization the same semantic must hold.
#[test]
fn test_weight_loosening_loosens_unreachable() {
    // Build the DWA (num_tsids / max_token unused but required)
    let mut dwa = DWA::new(256, 256);

    let s1 = dwa.add_state();

    // Transition from 0 → 1 only allows TSID 0
    dwa.add_transition(0, 1 as Label, s1, weight_from_item(0));

    // State 1 is final but only for TSIDs [5..=10]
    dwa.set_final_weight(s1, weight_from_range(5, 10));

    // Verify no word accepts before minimization
    let accept_before = dwa.eval_word(&[1 as Label]);
    assert!(
        accept_before.is_empty(),
        "Expected no acceptance before, got {:?}",
        accept_before
    );

    // Apply minimization (standalone fn, returns new DWA)
    let minimized = minimize::minimize(&dwa);

    // Verify semantics preserved after minimization
    let accept_after = minimized.eval_word(&[1 as Label]);
    assert!(
        accept_after.is_empty(),
        "Expected no acceptance after minimization, got {:?}",
        accept_after
    );
}
