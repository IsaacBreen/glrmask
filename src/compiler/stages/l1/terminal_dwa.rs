//! L1 terminal DWA: fast direct construction for terminals with max path length ≤ 1.
//!
//! Since L1 terminals never co-occur with another terminal in a single token,
//! the DWA can be built by walking each token from each state and checking
//! which terminal matches at the end.
//!
//! TODO: Extract L1-specific build_l1_fast logic from terminal_dwa/mod.rs.
