//! Parser-state labels.
//!
//! Parser-DWA input symbols are parser stack states.  The underlying automata
//! package uses signed `i32` labels because it also has default and internal
//! negative labels.  This helper is the one place where a raw automaton label
//! is interpreted as a parser-state id.

pub(crate) fn parser_state_label(label: i32, num_parser_states: u32) -> Option<u32> {
    if label >= 0 && (label as u32) < num_parser_states {
        Some(label as u32)
    } else {
        None
    }
}
