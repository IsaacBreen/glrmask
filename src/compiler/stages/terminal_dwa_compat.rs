//! Compatibility surface for the legacy terminal-DWA fallback path.
//!
//! The canonical terminal-DWA implementation lives under
//! `id_map_and_terminal_dwa/`. This module exists only for code paths that
//! already have an `InternalIdMap` and need the old non-split builder.
