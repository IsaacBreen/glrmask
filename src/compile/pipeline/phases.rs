//! Named compile phases.
//!
//! The compile pipeline should read like the paper's construction order.  Each
//! variant is a mathematical boundary: it transforms one named object into one
//! or more later named objects.

/// Coarse compile phases in dependency order.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(crate) enum CompilePhase {
    /// Normalize frontend output into the grammar normal form consumed by all later phases.
    ImportNormalize,
    /// Build the lexer/tokenizer DFA from grammar terminals.
    BuildTokenizer,
    /// Analyze the grammar into item sets, reductions, and table preconditions.
    AnalyzeGrammar,
    /// Build the GLR transition/action table.
    BuildGlrTable,
    /// Build terminal coloring and follow-exclusion data derived from the table/grammar.
    BuildTerminalGrammarFacts,
    /// Build the Terminal DWA over complete terminal strings.
    BuildTerminalDwa,
    /// Build the scan relation / CanMatch artifact for partial token scans.
    BuildScanRelation,
    /// Build stack-effect template DFAs for commit acceleration and Parser-DWA construction.
    BuildTemplates,
    /// Build the Parser DWA over parser-stack prefixes.
    BuildParserDwa,
    /// Reconcile internal ID spaces shared by Terminal DWA, Parser DWA, and CanMatch.
    ReconcileArtifact,
    /// Assemble and cache the runtime [`crate::Constraint`] artifact.
    FinalizeRuntime,
}

impl CompilePhase {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::ImportNormalize => "import_normalize",
            Self::BuildTokenizer => "build_tokenizer",
            Self::AnalyzeGrammar => "analyze_grammar",
            Self::BuildGlrTable => "build_glr_table",
            Self::BuildTerminalGrammarFacts => "build_terminal_grammar_facts",
            Self::BuildTerminalDwa => "build_terminal_dwa",
            Self::BuildScanRelation => "build_scan_relation",
            Self::BuildTemplates => "build_templates",
            Self::BuildParserDwa => "build_parser_dwa",
            Self::ReconcileArtifact => "reconcile_artifact",
            Self::FinalizeRuntime => "finalize_runtime",
        }
    }

    pub(crate) fn description(self) -> &'static str {
        match self {
            Self::ImportNormalize => "lower and normalize frontend grammar IR",
            Self::BuildTokenizer => "compile terminal regexes/literals into the lexer DFA",
            Self::AnalyzeGrammar => "derive parser item and reduction facts",
            Self::BuildGlrTable => "build the parser action/goto table",
            Self::BuildTerminalGrammarFacts => "derive terminal colors and disallowed follows",
            Self::BuildTerminalDwa => "compile complete-token terminal strings into Terminal DWA weights",
            Self::BuildScanRelation => "compile partial-token scan completions into CanMatch weights",
            Self::BuildTemplates => "compile stack-effect recognizers/template DFAs",
            Self::BuildParserDwa => "compile stack-prefix acceptance into Parser DWA weights",
            Self::ReconcileArtifact => "put all weighted artifacts in one internal coordinate system",
            Self::FinalizeRuntime => "assemble runtime caches and public constraint object",
        }
    }
}

/// The canonical phase order, useful for docs and future structured reports.
pub(crate) const COMPILE_PHASE_ORDER: &[CompilePhase] = &[
    CompilePhase::ImportNormalize,
    CompilePhase::BuildTokenizer,
    CompilePhase::AnalyzeGrammar,
    CompilePhase::BuildGlrTable,
    CompilePhase::BuildTerminalGrammarFacts,
    CompilePhase::BuildTerminalDwa,
    CompilePhase::BuildScanRelation,
    CompilePhase::BuildTemplates,
    CompilePhase::BuildParserDwa,
    CompilePhase::ReconcileArtifact,
    CompilePhase::FinalizeRuntime,
];
