//! Regression coverage for typed characterization-cycle declines.
//!
//! The syntactic right-recursion gate normally rejects cyclic grammars before a
//! table exists. At HEAD, bypassing that gate does not reliably preserve a
//! reduction cycle through normalization (the reduced 2633 fixture compiles
//! successfully), so these tests drive the same production fallible entry point
//! directly and verify that its error maps to a strategy decline without an
//! unwind.

use std::panic::{AssertUnwindSafe, catch_unwind};

use glrmask_parser_dwa::__private::templates::characterize::
    try_characterize_selected_terminals_for_terminal_count;

use crate::Error;
use crate::compiler::glr::table::{Action, testing::build_test_table};

fn cyclic_table() -> crate::compiler::glr::table::GLRTable {
    // The reduction crosses its pushed goto prefix and may repeat, so the
    // relation contains a genuine NtRereduce cycle.
    let actions = [
        vec![],
        vec![(0, Action::Reduce(0, 2)), (1, Action::Reduce(1, 2))],
        vec![],
        vec![(0, Action::Shift(4, false)), (1, Action::Shift(4, false))],
        vec![],
    ];
    let gotos = [
        vec![(0, (1, false)), (1, (1, false))],
        vec![],
        vec![(0, (3, false)), (1, (3, false))],
        vec![],
        vec![],
    ];
    build_test_table(
        5,
        2,
        &actions.iter().map(Vec::as_slice).collect::<Vec<_>>(),
        &gotos.iter().map(Vec::as_slice).collect::<Vec<_>>(),
    )
}

#[test]
fn production_characterization_cycle_is_a_typed_decline_without_unwinding() {
    let table = cyclic_table();
    let result = catch_unwind(AssertUnwindSafe(|| {
        try_characterize_selected_terminals_for_terminal_count(&table, 2, &[true, true])
    }))
    .expect("the fallible characterization path must not unwind");
    let error = result.expect_err("a cyclic relation must not be published");
    assert_eq!(
        error.failed_terminal(),
        0,
        "the lowest failing terminal is deterministic",
    );
    assert!(
        error.to_string().contains("reduction cycle"),
        "the established diagnostic text must be preserved: {error}",
    );
    let mapped = Error::classify_template_compile_message(error.to_string());
    assert!(
        mapped.is_strategy_decline(),
        "a characterization cycle must map to a strategy decline, got: {mapped}",
    );
}
