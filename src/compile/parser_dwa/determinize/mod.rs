//! Weighted determinization phases for Parser-DWA construction.
//!
//! This submodule is split by mathematical role rather than by helper size:
//!
//! - `outgoing`: recover possible parser-state labels from NWA supports;
//! - `epsilon`: local weighted epsilon closure;
//! - `support`: first determinization, preserving source-NWA supports;
//! - `fallback`: second determinization, making default fallback semantics
//!   explicit.

mod epsilon;
mod fallback;
mod outgoing;
mod support;

pub(crate) use fallback::determinize_parser_dwa_with_fallbacks;
pub(crate) use outgoing::build_possible_outgoing_ids_by_state;
pub(crate) use support::determinize_with_supports;
