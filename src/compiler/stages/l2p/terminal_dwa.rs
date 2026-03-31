//! L2+ terminal DWA: full NWA-based construction for terminals with path length ≥ 2.
//!
//! Uses the existing NWA build → postprocess → determinize → minimize pipeline,
//! but only for L2+ terminal groups.
//!
//! TODO: Extract L2+-specific logic from terminal_dwa/mod.rs.
