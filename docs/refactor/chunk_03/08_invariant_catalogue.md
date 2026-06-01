# Chunk 03 invariant catalogue


### Invariant 1: ImportNormalize has a single denotation

`ImportNormalize` produces `prepared GrammarDef` from `frontend grammar`.  Its purpose is language-preserving grammar normal form.  The key rule is: Do not inspect token bytes here. Do not decide artifact coordinate systems here. This is a source-shape phase.

A maintainer may optimize the algorithm, split the implementation into more files, or recover parallelism, but must not change the denotation without renaming the phase and documenting the new contract.  If a later change needs extra data, prefer adding a field to a typed context/output struct over smuggling the data through a local variable in the orchestrator.

**Acceptable changes:** local optimization, more explicit profile counters, smaller helper functions, stronger precondition checks, clearer names.

**Suspicious changes:** direct environment reads, direct stderr prints, runtime cache fields appearing in non-final phases, Terminal-DWA equivalence assumptions leaking into scan relation, parser implementation details becoming public terminology.


### Invariant 2: BuildTokenizer has a single denotation

`BuildTokenizer` produces `Tokenizer` from `prepared GrammarDef terminals`.  Its purpose is byte-level recognition of grammar terminals.  The key rule is: The tokenizer is grammar-only. Vocabulary appears later when token bytes are classified.

A maintainer may optimize the algorithm, split the implementation into more files, or recover parallelism, but must not change the denotation without renaming the phase and documenting the new contract.  If a later change needs extra data, prefer adding a field to a typed context/output struct over smuggling the data through a local variable in the orchestrator.

**Acceptable changes:** local optimization, more explicit profile counters, smaller helper functions, stronger precondition checks, clearer names.

**Suspicious changes:** direct environment reads, direct stderr prints, runtime cache fields appearing in non-final phases, Terminal-DWA equivalence assumptions leaking into scan relation, parser implementation details becoming public terminology.


### Invariant 3: AnalyzeGrammar has a single denotation

`AnalyzeGrammar` produces `AnalyzedGrammar` from `prepared GrammarDef productions`.  Its purpose is parser-side semantic facts.  The key rule is: Keep this separate from specific runtime masking decisions. The parser is a provider of stack-effect facts.

A maintainer may optimize the algorithm, split the implementation into more files, or recover parallelism, but must not change the denotation without renaming the phase and documenting the new contract.  If a later change needs extra data, prefer adding a field to a typed context/output struct over smuggling the data through a local variable in the orchestrator.

**Acceptable changes:** local optimization, more explicit profile counters, smaller helper functions, stronger precondition checks, clearer names.

**Suspicious changes:** direct environment reads, direct stderr prints, runtime cache fields appearing in non-final phases, Terminal-DWA equivalence assumptions leaking into scan relation, parser implementation details becoming public terminology.


### Invariant 4: BuildGlrTable has a single denotation

`BuildGlrTable` produces `GLRTable` from `AnalyzedGrammar`.  Its purpose is implementation witness for stack evolution.  The key rule is: Do not let public exposition suggest the method is only LR. The table is one way to obtain stack effects.

A maintainer may optimize the algorithm, split the implementation into more files, or recover parallelism, but must not change the denotation without renaming the phase and documenting the new contract.  If a later change needs extra data, prefer adding a field to a typed context/output struct over smuggling the data through a local variable in the orchestrator.

**Acceptable changes:** local optimization, more explicit profile counters, smaller helper functions, stronger precondition checks, clearer names.

**Suspicious changes:** direct environment reads, direct stderr prints, runtime cache fields appearing in non-final phases, Terminal-DWA equivalence assumptions leaking into scan relation, parser implementation details becoming public terminology.


### Invariant 5: BuildTerminalGrammarFacts has a single denotation

`BuildTerminalGrammarFacts` produces `TerminalColoring + disallowed follows` from `GLRTable + AnalyzedGrammar`.  Its purpose is terminal coexistence and legality facts.  The key rule is: These facts help Terminal-DWA construction without changing Terminal-DWA denotation.

A maintainer may optimize the algorithm, split the implementation into more files, or recover parallelism, but must not change the denotation without renaming the phase and documenting the new contract.  If a later change needs extra data, prefer adding a field to a typed context/output struct over smuggling the data through a local variable in the orchestrator.

**Acceptable changes:** local optimization, more explicit profile counters, smaller helper functions, stronger precondition checks, clearer names.

**Suspicious changes:** direct environment reads, direct stderr prints, runtime cache fields appearing in non-final phases, Terminal-DWA equivalence assumptions leaking into scan relation, parser implementation details becoming public terminology.


### Invariant 6: BuildTerminalDwa has a single denotation

`BuildTerminalDwa` produces `MappedArtifact<DWA>` from `Tokenizer + Vocab + terminal facts`.  Its purpose is complete terminal sequence to lexer-state/token mask.  The key rule is: This is the paper Terminal DWA. Preserve the name and keep complete-vs-partial scan distinction explicit.

A maintainer may optimize the algorithm, split the implementation into more files, or recover parallelism, but must not change the denotation without renaming the phase and documenting the new contract.  If a later change needs extra data, prefer adding a field to a typed context/output struct over smuggling the data through a local variable in the orchestrator.

**Acceptable changes:** local optimization, more explicit profile counters, smaller helper functions, stronger precondition checks, clearer names.

**Suspicious changes:** direct environment reads, direct stderr prints, runtime cache fields appearing in non-final phases, Terminal-DWA equivalence assumptions leaking into scan relation, parser implementation details becoming public terminology.


### Invariant 7: BuildScanRelation has a single denotation

`BuildScanRelation` produces `ScanRelationComputation` from `Tokenizer + Vocab`.  Its purpose is partial byte-fragment completability.  The key rule is: Never reuse Terminal-DWA quotienting as a correctness proof here.

A maintainer may optimize the algorithm, split the implementation into more files, or recover parallelism, but must not change the denotation without renaming the phase and documenting the new contract.  If a later change needs extra data, prefer adding a field to a typed context/output struct over smuggling the data through a local variable in the orchestrator.

**Acceptable changes:** local optimization, more explicit profile counters, smaller helper functions, stronger precondition checks, clearer names.

**Suspicious changes:** direct environment reads, direct stderr prints, runtime cache fields appearing in non-final phases, Terminal-DWA equivalence assumptions leaking into scan relation, parser implementation details becoming public terminology.


### Invariant 8: BuildTemplates has a single denotation

`BuildTemplates` produces `Templates + TemplateDfasByTerminal` from `GLRTable + AnalyzedGrammar`.  Its purpose is stack-effect recognizers.  The key rule is: Templates are parser-abstract; avoid over-explaining LR mechanics.

A maintainer may optimize the algorithm, split the implementation into more files, or recover parallelism, but must not change the denotation without renaming the phase and documenting the new contract.  If a later change needs extra data, prefer adding a field to a typed context/output struct over smuggling the data through a local variable in the orchestrator.

**Acceptable changes:** local optimization, more explicit profile counters, smaller helper functions, stronger precondition checks, clearer names.

**Suspicious changes:** direct environment reads, direct stderr prints, runtime cache fields appearing in non-final phases, Terminal-DWA equivalence assumptions leaking into scan relation, parser implementation details becoming public terminology.


### Invariant 9: BuildParserDwa has a single denotation

`BuildParserDwa` produces `Parser DWA` from `Terminal DWA + Templates + parser facts`.  Its purpose is stack-prefix to token mask automaton.  The key rule is: The Parser DWA is the runtime Mask object over active stack prefixes.

A maintainer may optimize the algorithm, split the implementation into more files, or recover parallelism, but must not change the denotation without renaming the phase and documenting the new contract.  If a later change needs extra data, prefer adding a field to a typed context/output struct over smuggling the data through a local variable in the orchestrator.

**Acceptable changes:** local optimization, more explicit profile counters, smaller helper functions, stronger precondition checks, clearer names.

**Suspicious changes:** direct environment reads, direct stderr prints, runtime cache fields appearing in non-final phases, Terminal-DWA equivalence assumptions leaking into scan relation, parser implementation details becoming public terminology.


### Invariant 10: ReconcileArtifact has a single denotation

`ReconcileArtifact` produces `ReconciledArtifacts` from `Parser DWA + CanMatch + Terminal DWA maps`.  Its purpose is one shared internal coordinate system.  The key rule is: This phase is a proof that all final weights speak the same ID language.

A maintainer may optimize the algorithm, split the implementation into more files, or recover parallelism, but must not change the denotation without renaming the phase and documenting the new contract.  If a later change needs extra data, prefer adding a field to a typed context/output struct over smuggling the data through a local variable in the orchestrator.

**Acceptable changes:** local optimization, more explicit profile counters, smaller helper functions, stronger precondition checks, clearer names.

**Suspicious changes:** direct environment reads, direct stderr prints, runtime cache fields appearing in non-final phases, Terminal-DWA equivalence assumptions leaking into scan relation, parser implementation details becoming public terminology.


### Invariant 11: FinalizeRuntime has a single denotation

`FinalizeRuntime` produces `Constraint` from `ReconciledArtifacts + Vocab`.  Its purpose is runtime representation.  The key rule is: Dense caches and sparse caches are derived representations only.

A maintainer may optimize the algorithm, split the implementation into more files, or recover parallelism, but must not change the denotation without renaming the phase and documenting the new contract.  If a later change needs extra data, prefer adding a field to a typed context/output struct over smuggling the data through a local variable in the orchestrator.

**Acceptable changes:** local optimization, more explicit profile counters, smaller helper functions, stronger precondition checks, clearer names.

**Suspicious changes:** direct environment reads, direct stderr prints, runtime cache fields appearing in non-final phases, Terminal-DWA equivalence assumptions leaking into scan relation, parser implementation details becoming public terminology.
