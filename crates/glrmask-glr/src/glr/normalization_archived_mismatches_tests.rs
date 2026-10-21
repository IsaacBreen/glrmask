//! Frozen historical normalization mismatch corpus from the 2026-10-19 audit.
//! Original input source: case-inventory.json, SHA256 38e4e9562e2fcd62e34c8a352e12238bb44521145958f8449fb01996289e610e.
//! Rules use the saved encoding: negative -1..-3 are terminal IDs 0..2.
//! Each archived original and minimized grammar is compared to an independent
//! finite fixed-point language, excluding epsilon (tracked separately by caller).

use super::*;

#[test]
fn archived_normalization_case_db0b1ca43bd81dae() {
    // Saved mismatch seed 4735; historical ID db0b1ca43bd81dae, bound 4.
    let input = vec![
        rule(0, vec![]),
        rule(0, vec![]),
        rule(0, vec![]),
        rule(1, vec![]),
        rule(2, vec![nt(4), nt(0), nt(5), nt(1)]),
        rule(2, vec![t(0), t(1), nt(2)]),
        rule(2, vec![]),
        rule(3, vec![t(1), nt(2), t(2), nt(1)]),
        rule(4, vec![nt(0), t(0), t(0), nt(2)]),
        rule(4, vec![nt(0), nt(5), nt(1), t(2)]),
        rule(4, vec![t(0), t(1)]),
        rule(5, vec![]),
    ];
    let minimized = vec![
        rule(0, vec![]),
        rule(2, vec![nt(4)]),
        rule(2, vec![t(0), nt(2)]),
        rule(5, vec![]),
    ];
    for (kind, rules) in [("original", input), ("minimized", minimized)] {
        let mut expected = bounded_language(&rules, 5, 4);
        expected.remove(&Vec::new());
        let normalized = compare_ordered(&rules, 5);
        let observed = bounded_language(&normalized, 5, 4);
        assert_eq!(observed, expected, "seed 4735, db0b1ca43bd81dae, {kind}");
    }
}

#[test]
fn archived_normalization_case_e33ed367492484a8() {
    // Saved mismatch seed 5019; historical ID e33ed367492484a8, bound 4.
    let input = vec![
        rule(0, vec![t(2), t(2), nt(0)]),
        rule(1, vec![t(0)]),
        rule(2, vec![]),
    ];
    let minimized = vec![
        rule(0, vec![t(2), nt(0)]),
        rule(1, vec![t(0)]),
    ];
    for (kind, rules) in [("original", input), ("minimized", minimized)] {
        let mut expected = bounded_language(&rules, 2, 4);
        expected.remove(&Vec::new());
        let normalized = compare_ordered(&rules, 2);
        let observed = bounded_language(&normalized, 2, 4);
        assert_eq!(observed, expected, "seed 5019, e33ed367492484a8, {kind}");
    }
}

#[test]
fn archived_normalization_case_affe826d17cd1abc() {
    // Saved mismatch seed 6123; historical ID affe826d17cd1abc, bound 4.
    let input = vec![
        rule(0, vec![t(0)]),
        rule(0, vec![t(1), nt(0)]),
        rule(0, vec![nt(1), nt(1), nt(1), nt(0)]),
        rule(0, vec![t(0), nt(1), nt(1), nt(1)]),
        rule(1, vec![]),
    ];
    let minimized = vec![
        rule(0, vec![t(1), nt(0)]),
        rule(1, vec![]),
    ];
    for (kind, rules) in [("original", input), ("minimized", minimized)] {
        let mut expected = bounded_language(&rules, 1, 4);
        expected.remove(&Vec::new());
        let normalized = compare_ordered(&rules, 1);
        let observed = bounded_language(&normalized, 1, 4);
        assert_eq!(observed, expected, "seed 6123, affe826d17cd1abc, {kind}");
    }
}

#[test]
fn archived_normalization_case_7d80312a23fd5681() {
    // Saved mismatch seed 6205; historical ID 7d80312a23fd5681, bound 4.
    let input = vec![
        rule(0, vec![t(2), nt(1), t(1), t(1)]),
        rule(0, vec![t(2), nt(0), nt(3), nt(3)]),
        rule(1, vec![]),
        rule(1, vec![]),
        rule(1, vec![t(0)]),
        rule(2, vec![nt(3), nt(3), t(1)]),
        rule(2, vec![nt(2), t(1)]),
        rule(2, vec![nt(1), t(2), nt(2), t(2)]),
        rule(3, vec![]),
    ];
    let minimized = vec![
        rule(0, vec![t(2), nt(0), nt(3), nt(3)]),
        rule(2, vec![nt(3), nt(3), t(1)]),
        rule(3, vec![]),
    ];
    for (kind, rules) in [("original", input), ("minimized", minimized)] {
        let mut expected = bounded_language(&rules, 3, 4);
        expected.remove(&Vec::new());
        let normalized = compare_ordered(&rules, 3);
        let observed = bounded_language(&normalized, 3, 4);
        assert_eq!(observed, expected, "seed 6205, 7d80312a23fd5681, {kind}");
    }
}

#[test]
fn archived_normalization_case_bf65071194ce1504() {
    // Saved mismatch seed 10667; historical ID bf65071194ce1504, bound 4.
    let input = vec![
        rule(0, vec![nt(2), nt(0), nt(0)]),
        rule(0, vec![t(2), nt(2), t(2)]),
        rule(0, vec![nt(1), nt(1), t(0), nt(1)]),
        rule(1, vec![nt(1)]),
        rule(1, vec![t(1), nt(1)]),
        rule(2, vec![]),
        rule(2, vec![]),
        rule(3, vec![]),
    ];
    let minimized = vec![
        rule(0, vec![nt(0), nt(0)]),
        rule(0, vec![t(2), nt(2)]),
        rule(2, vec![]),
    ];
    for (kind, rules) in [("original", input), ("minimized", minimized)] {
        let mut expected = bounded_language(&rules, 3, 4);
        expected.remove(&Vec::new());
        let normalized = compare_ordered(&rules, 3);
        let observed = bounded_language(&normalized, 3, 4);
        assert_eq!(observed, expected, "seed 10667, bf65071194ce1504, {kind}");
    }
}

#[test]
fn archived_normalization_case_e81c830906eaf23c() {
    // Saved mismatch seed 12119; historical ID e81c830906eaf23c, bound 4.
    let input = vec![
        rule(0, vec![nt(0), t(0)]),
        rule(0, vec![t(2)]),
        rule(1, vec![nt(0), nt(2), nt(0)]),
        rule(2, vec![]),
        rule(2, vec![]),
        rule(2, vec![]),
        rule(2, vec![t(1), t(2), nt(2)]),
        rule(3, vec![]),
    ];
    let minimized = vec![
        rule(2, vec![t(1), nt(2)]),
    ];
    for (kind, rules) in [("original", input), ("minimized", minimized)] {
        let mut expected = bounded_language(&rules, 3, 4);
        expected.remove(&Vec::new());
        let normalized = compare_ordered(&rules, 3);
        let observed = bounded_language(&normalized, 3, 4);
        assert_eq!(observed, expected, "seed 12119, e81c830906eaf23c, {kind}");
    }
}

#[test]
fn archived_normalization_case_245271340dba9d6e() {
    // Saved mismatch seed 15553; historical ID 245271340dba9d6e, bound 4.
    let input = vec![
        rule(0, vec![nt(0), nt(0), t(2), nt(0)]),
        rule(0, vec![nt(0), t(0)]),
        rule(0, vec![nt(0)]),
        rule(0, vec![nt(1), nt(1), t(0)]),
        rule(1, vec![]),
        rule(1, vec![nt(1), nt(1)]),
    ];
    let minimized = vec![
        rule(0, vec![nt(0), nt(0)]),
        rule(0, vec![nt(1), nt(1), t(0)]),
        rule(1, vec![]),
    ];
    for (kind, rules) in [("original", input), ("minimized", minimized)] {
        let mut expected = bounded_language(&rules, 1, 4);
        expected.remove(&Vec::new());
        let normalized = compare_ordered(&rules, 1);
        let observed = bounded_language(&normalized, 1, 4);
        assert_eq!(observed, expected, "seed 15553, 245271340dba9d6e, {kind}");
    }
}

#[test]
fn archived_normalization_case_d9b06aad8ef45a25() {
    // Saved mismatch seed 17359; historical ID d9b06aad8ef45a25, bound 4.
    let input = vec![
        rule(0, vec![nt(0), nt(0), t(2), t(1)]),
        rule(0, vec![nt(1), nt(0)]),
        rule(1, vec![]),
        rule(1, vec![t(0), nt(1)]),
        rule(1, vec![]),
        rule(2, vec![]),
    ];
    let minimized = vec![
        rule(1, vec![]),
        rule(1, vec![t(0), nt(1)]),
    ];
    for (kind, rules) in [("original", input), ("minimized", minimized)] {
        let mut expected = bounded_language(&rules, 2, 4);
        expected.remove(&Vec::new());
        let normalized = compare_ordered(&rules, 2);
        let observed = bounded_language(&normalized, 2, 4);
        assert_eq!(observed, expected, "seed 17359, d9b06aad8ef45a25, {kind}");
    }
}

#[test]
fn archived_normalization_case_96bf5e9599705ccc() {
    // Saved mismatch seed 23190; historical ID 96bf5e9599705ccc, bound 6.
    let input = vec![
        rule(0, vec![t(0)]),
        rule(1, vec![t(2)]),
        rule(2, vec![t(1)]),
        rule(3, vec![t(0)]),
        rule(4, vec![t(0)]),
        rule(0, vec![nt(3), nt(3), nt(1), t(2)]),
        rule(0, vec![t(0), t(1), nt(3)]),
        rule(0, vec![t(2), nt(2), nt(2), nt(3)]),
        rule(0, vec![t(1), nt(2), nt(0)]),
        rule(1, vec![nt(3), nt(1)]),
        rule(1, vec![nt(0), t(2), nt(3)]),
        rule(1, vec![]),
        rule(1, vec![]),
        rule(2, vec![nt(0), nt(1), nt(3)]),
        rule(2, vec![t(2)]),
        rule(2, vec![nt(0), t(1)]),
        rule(3, vec![]),
        rule(3, vec![]),
        rule(3, vec![t(0), t(1), nt(0), t(0)]),
        rule(3, vec![nt(4)]),
        rule(4, vec![t(0)]),
    ];
    let minimized = vec![
        rule(0, vec![t(0)]),
        rule(3, vec![t(0)]),
        rule(0, vec![nt(3), nt(1), t(2)]),
        rule(1, vec![nt(3), nt(1)]),
        rule(1, vec![nt(0), t(2)]),
        rule(1, vec![]),
        rule(2, vec![nt(1)]),
        rule(3, vec![]),
    ];
    for (kind, rules) in [("original", input), ("minimized", minimized)] {
        let mut expected = bounded_language(&rules, 2, 6);
        expected.remove(&Vec::new());
        let normalized = compare_ordered(&rules, 2);
        let observed = bounded_language(&normalized, 2, 6);
        assert_eq!(observed, expected, "seed 23190, 96bf5e9599705ccc, {kind}");
    }
}

#[test]
fn archived_normalization_case_da40bbce5b04e3fd() {
    // Saved mismatch seed 25217; historical ID da40bbce5b04e3fd, bound 4.
    let input = vec![
        rule(0, vec![]),
        rule(0, vec![t(0), t(2)]),
        rule(1, vec![nt(2)]),
        rule(1, vec![nt(0), nt(3), nt(1)]),
        rule(1, vec![nt(4), nt(0), nt(2), nt(1)]),
        rule(2, vec![nt(1), nt(4), nt(3), t(0)]),
        rule(3, vec![nt(1), nt(4), nt(4)]),
        rule(3, vec![nt(0), nt(2)]),
        rule(3, vec![nt(2), t(2), t(1), nt(3)]),
        rule(3, vec![t(0), nt(0), t(1), t(0)]),
        rule(4, vec![t(1)]),
        rule(5, vec![]),
    ];
    let minimized = vec![
        rule(0, vec![]),
        rule(1, vec![nt(3), nt(1)]),
        rule(1, vec![nt(4)]),
        rule(3, vec![t(0)]),
    ];
    for (kind, rules) in [("original", input), ("minimized", minimized)] {
        let mut expected = bounded_language(&rules, 5, 4);
        expected.remove(&Vec::new());
        let normalized = compare_ordered(&rules, 5);
        let observed = bounded_language(&normalized, 5, 4);
        assert_eq!(observed, expected, "seed 25217, da40bbce5b04e3fd, {kind}");
    }
}
