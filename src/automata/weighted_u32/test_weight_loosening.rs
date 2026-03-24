//! Regression tests for weight loosening.

use super::dwa::DWA;
use super::minimize;
use super::nwa::Label;
use super::test_support::{weight_from_item, weight_from_range};

#[test]
fn test_minimize_preserves_unreachable_weight_rejection() {
    let mut dwa = DWA::new(256, 256);

    let s1 = dwa.add_state();

    dwa.add_transition(0, 1 as Label, s1, weight_from_item(0));
    dwa.set_final_weight(s1, weight_from_range(5, 10));

    let accept_before = dwa.eval_word(&[1 as Label]);
    assert!(
        accept_before.is_empty(),
        "Expected no acceptance before, got {:?}",
        accept_before
    );

    let minimized = minimize::minimize(&dwa);

    let accept_after = minimized.eval_word(&[1 as Label]);
    assert!(
        accept_after.is_empty(),
        "Expected no acceptance after minimization, got {:?}",
        accept_after
    );
}
