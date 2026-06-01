# Chunk 03 dependency graph and deferred parallelism

## Dependency graph

```text
OwnedCompileInput
  |
  v
ImportNormalize
  |
  v
PreparedCompileInput
  |
  +--> BuildTokenizer ---------------------+
  |                                        |
  +--> AnalyzeGrammar --> BuildGlrTable ---+--> GrammarAnalysisOutput
                         |                 |
                         v                 |
                  BuildTerminalGrammarFacts+
                                           |
                                           v
                             TerminalScanSupport
                                           |
                 +-------------------------+-------------------------+
                 |                                                   |
                 v                                                   v
          BuildTerminalDwa                                    BuildScanRelation
                 |                                                   |
                 +-------------------- TerminalAndScanOutput --------+
                                           |
GrammarAnalysisOutput ---------------------+------------------+
                                                              |
                                                              v
                                                       BuildTemplates
                                                              |
                                                              v
                                     TemplateOutput + TerminalAndScanOutput
                                                              |
                                                              v
                                          BuildParserDwa / ReconcileArtifact
                                                              |
                                                              v
                                                   ReconciledArtifacts
                                                              |
                                                              v
                                                       FinalizeRuntime
                                                              |
                                                              v
                                                        Constraint
```

## Why this graph matters

The old file let concurrency obscure dependency.  The new code should make dependencies obvious even when later chunks recover parallelism.  If we add a `rayon::join`, it should be because two named phase outputs are independent, not because two blocks happened to fit in one closure.

## Deferred parallelism plan

The old implementation built terminal/scan artifacts and templates in parallel while both closures contributed to one profile object.  The refactor currently prioritizes clarity.  To restore parallelism cleanly, change phase functions so they return local profile deltas.

Target shape:

```rust
struct PhaseOutput<T> {
    value: T,
    profile_delta: CompilePhaseProfile,
}

let (terminal_scan, templates) = rayon::join(
    || build_terminal_dwa_and_scan_relation(...),
    || build_templates(...),
);
profile += terminal_scan.profile_delta;
profile += templates.profile_delta;
```

This makes parallelism a property of the phase graph rather than an accident of mutable local variables.

## Parallel groups that are mathematically safe

1. **Tokenizer vs grammar analysis.** Tokenizer construction uses terminal regex/literal definitions. Grammar analysis uses grammar productions. They share the prepared grammar but neither consumes the other's output until terminal facts are assembled.
2. **Terminal DWA vs scan relation.** They share tokenizer/vocab facts and classification support, but their outputs are independent before reconciliation.
3. **Templates vs Terminal-DWA/scan construction.** Templates depend on parser stack-effect facts. Terminal-DWA/scan construction depends on tokenizer/vocab facts plus terminal grammar facts. They join at Parser-DWA/reconciliation.
4. **Parser-DWA construction vs some compaction planning.** Some reconciliation modes can plan compaction while Parser DWA builds over pre-compact IDs, as the old implementation did.

## Parallel groups that are not safe without more proof

1. **Terminal-DWA equivalence reused for CanMatch.** Not safe. The denotations differ.
2. **Final runtime cache rebuild before reconciliation.** Not safe. Runtime caches need final internal token IDs.
3. **Parser-DWA build before template construction.** Not safe. Parser-DWA construction consumes templates.
4. **CanMatch final materialization before deciding shared ID space.** Not safe unless the artifact records enough mapping to be safely reconciled later.
