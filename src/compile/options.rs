//! Environment-backed compile options.
//!
//! This is an intermediate publication-cleanup boundary.  The public
//! [`crate::CompileOptions`] type exists, but most historical tuning still comes
//! from environment variables.  The important structural change is that the
//! compile pipeline no longer asks `std::env` questions directly; it asks this
//! module for named compile decisions.

/// Interpret the usual boolean environment spellings.
pub(crate) fn env_flag_enabled(name: &str) -> bool {
    std::env::var(name)
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            !matches!(normalized.as_str(), "" | "0" | "false" | "no" | "off")
        })
        .unwrap_or(false)
}

/// Interpret the usual boolean environment spellings, defaulting to true.
pub(crate) fn env_flag_enabled_by_default(name: &str) -> bool {
    std::env::var(name)
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            !matches!(normalized.as_str(), "" | "0" | "false" | "no" | "off")
        })
        .unwrap_or(true)
}

/// Whether to compact the scan-relation / CanMatch artifact before ID-space reconciliation.
pub(crate) fn compact_can_match_before_reconcile_enabled() -> bool {
    env_flag_enabled_by_default("GLRMASK_COMPACT_CAN_MATCH_BEFORE_RECONCILE")
}

/// Whether the tokenizer build should emit per-terminal detail lines.
pub(crate) fn tokenizer_detail_profile_enabled() -> bool {
    std::env::var_os("GLRMASK_PROFILE_TOKENIZER_DETAIL").is_some()
}

/// How to reconcile the Terminal DWA, Parser DWA, and scan-relation CanMatch artifact.
///
/// The choices differ only in where the shared internal `(lexer-state, token)`
/// equivalence space is introduced and compacted.  The mathematical output is
/// the same: Parser-DWA weights and CanMatch weights must end in one common
/// internal coordinate system.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DwaCanMatchMode {
    /// Reconcile Terminal DWA with CanMatch; build Parser DWA over the reconciled map.
    TerminalReconcile,
    /// Reconcile Terminal DWA with CanMatch and compact that joint space.
    TerminalReconcileAndCompact,
    /// Reconcile Terminal DWA with CanMatch, then compact Parser DWA with CanMatch.
    TerminalReconcileAndParserCompact,
    /// Compact Terminal DWA with CanMatch before Parser DWA, then compact Parser DWA with CanMatch.
    TerminalReconcileAndTerminalCompactAndParserCompact,
    /// Build Parser DWA first, then reconcile Parser DWA with CanMatch.
    ParserReconcile,
    /// Build Parser DWA first, then reconcile and compact Parser DWA with CanMatch.
    ParserReconcileAndCompact,
}

impl DwaCanMatchMode {
    pub(crate) fn does_terminal_reconcile(self) -> bool {
        matches!(
            self,
            Self::TerminalReconcile
                | Self::TerminalReconcileAndCompact
                | Self::TerminalReconcileAndParserCompact
                | Self::TerminalReconcileAndTerminalCompactAndParserCompact
        )
    }

    pub(crate) fn does_terminal_compact(self) -> bool {
        matches!(
            self,
            Self::TerminalReconcileAndCompact
                | Self::TerminalReconcileAndTerminalCompactAndParserCompact
        )
    }

    pub(crate) fn does_parser_compact(self) -> bool {
        matches!(
            self,
            Self::TerminalReconcileAndParserCompact
                | Self::TerminalReconcileAndTerminalCompactAndParserCompact
                | Self::ParserReconcileAndCompact
        )
    }
}

/// Resolve the DWA/CanMatch reconciliation strategy from historical env vars.
pub(crate) fn dwa_can_match_mode() -> DwaCanMatchMode {
    match std::env::var("GLRMASK_DWA_CAN_MATCH_MODE")
        .or_else(|_| std::env::var("GLRMASK_PARSER_DWA_CAN_MATCH_COMPACTION"))
    {
        Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
            "" | "0" | "false" | "no" | "off" | "terminal" | "term"
            | "term_can_match_reconcile" | "terminal_can_match_reconcile" => {
                DwaCanMatchMode::TerminalReconcile
            }
            "term_compact" | "terminal_compact" | "term_can_match_compact"
            | "terminal_can_match_compact" | "term_can_match_reconcile_compact"
            | "terminal_can_match_reconcile_compact" => {
                DwaCanMatchMode::TerminalReconcileAndCompact
            }
            "parser_compact" | "term_parser_compact" | "terminal_parser_compact"
            | "term_can_match_reconcile_parser_can_match_compact"
            | "terminal_can_match_reconcile_parser_can_match_compact" => {
                DwaCanMatchMode::TerminalReconcileAndParserCompact
            }
            "both" | "1" | "true" | "yes" | "on" | "term_and_parser_compact"
            | "terminal_and_parser_compact" | "term_can_match_compact_parser_can_match_compact"
            | "terminal_can_match_compact_parser_can_match_compact" => {
                DwaCanMatchMode::TerminalReconcileAndTerminalCompactAndParserCompact
            }
            "parser" | "only" | "parser_only" | "replace" | "parser_can_match_reconcile" => {
                DwaCanMatchMode::ParserReconcile
            }
            "parser_can_match_compact" | "parser_reconcile_compact"
            | "parser_can_match_reconcile_compact" => DwaCanMatchMode::ParserReconcileAndCompact,
            _ => DwaCanMatchMode::TerminalReconcile,
        },
        Err(_) => {
            // Parser-side CanMatch compaction remains available via
            // `GLRMASK_DWA_CAN_MATCH_MODE=both` and parser compact modes, but it is
            // not the default because large schemas can pay several extra compile
            // seconds for small artifact-size wins.
            DwaCanMatchMode::TerminalReconcileAndCompact
        }
    }
}

/// Resolve the compile thread count used by the internal rayon pool.
pub(crate) fn compile_thread_count() -> Option<usize> {
    if let Some(value) = std::env::var("GLRMASK_COMPILE_THREADS")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|&value| value > 0)
    {
        return Some(value);
    }

    if std::env::var_os("RAYON_NUM_THREADS").is_some() {
        return None;
    }

    #[cfg(target_os = "macos")]
    {
        return std::thread::available_parallelism()
            .ok()
            .map(|parallelism| parallelism.get().min(10))
            .filter(|&value| value > 1);
    }

    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}
