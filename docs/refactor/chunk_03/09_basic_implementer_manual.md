# Chunk 03 basic implementer manual

This file is written for someone who can edit files and follow instructions but does not know the compiler deeply.


## Task 1: verify `ImportNormalize`

1. Find the module that implements `ImportNormalize`.
2. Confirm it consumes `frontend grammar`.
3. Confirm it produces `prepared GrammarDef`.
4. Read the module comment and check that it says: language-preserving grammar normal form.
5. Search the module for `std::env`.  If found, move the decision to `src/compile/options.rs` unless the module is itself `options.rs`.
6. Search the module for `eprintln!`.  If found, move rendering to `src/compile/profiling.rs` unless the module is explicitly a profile sink.
7. Search the module for `Constraint {`.  Only `FinalizeRuntime` should initialize the runtime constraint.
8. Search the module for words like `possible_matches`, `pm`, `l1`, `l2p`, or `mask_game`.  Replace historical names with paper-aligned names unless the old word appears only in a compatibility note.
9. Check the output type.  If the phase returns a tuple of more than two conceptual objects, consider creating a named struct.
10. Add or update one doc paragraph explaining why the phase is needed.

Reason: Do not inspect token bytes here. Do not decide artifact coordinate systems here. This is a source-shape phase.

## Task 2: verify `BuildTokenizer`

1. Find the module that implements `BuildTokenizer`.
2. Confirm it consumes `prepared GrammarDef terminals`.
3. Confirm it produces `Tokenizer`.
4. Read the module comment and check that it says: byte-level recognition of grammar terminals.
5. Search the module for `std::env`.  If found, move the decision to `src/compile/options.rs` unless the module is itself `options.rs`.
6. Search the module for `eprintln!`.  If found, move rendering to `src/compile/profiling.rs` unless the module is explicitly a profile sink.
7. Search the module for `Constraint {`.  Only `FinalizeRuntime` should initialize the runtime constraint.
8. Search the module for words like `possible_matches`, `pm`, `l1`, `l2p`, or `mask_game`.  Replace historical names with paper-aligned names unless the old word appears only in a compatibility note.
9. Check the output type.  If the phase returns a tuple of more than two conceptual objects, consider creating a named struct.
10. Add or update one doc paragraph explaining why the phase is needed.

Reason: The tokenizer is grammar-only. Vocabulary appears later when token bytes are classified.

## Task 3: verify `AnalyzeGrammar`

1. Find the module that implements `AnalyzeGrammar`.
2. Confirm it consumes `prepared GrammarDef productions`.
3. Confirm it produces `AnalyzedGrammar`.
4. Read the module comment and check that it says: parser-side semantic facts.
5. Search the module for `std::env`.  If found, move the decision to `src/compile/options.rs` unless the module is itself `options.rs`.
6. Search the module for `eprintln!`.  If found, move rendering to `src/compile/profiling.rs` unless the module is explicitly a profile sink.
7. Search the module for `Constraint {`.  Only `FinalizeRuntime` should initialize the runtime constraint.
8. Search the module for words like `possible_matches`, `pm`, `l1`, `l2p`, or `mask_game`.  Replace historical names with paper-aligned names unless the old word appears only in a compatibility note.
9. Check the output type.  If the phase returns a tuple of more than two conceptual objects, consider creating a named struct.
10. Add or update one doc paragraph explaining why the phase is needed.

Reason: Keep this separate from specific runtime masking decisions. The parser is a provider of stack-effect facts.

## Task 4: verify `BuildGlrTable`

1. Find the module that implements `BuildGlrTable`.
2. Confirm it consumes `AnalyzedGrammar`.
3. Confirm it produces `GLRTable`.
4. Read the module comment and check that it says: implementation witness for stack evolution.
5. Search the module for `std::env`.  If found, move the decision to `src/compile/options.rs` unless the module is itself `options.rs`.
6. Search the module for `eprintln!`.  If found, move rendering to `src/compile/profiling.rs` unless the module is explicitly a profile sink.
7. Search the module for `Constraint {`.  Only `FinalizeRuntime` should initialize the runtime constraint.
8. Search the module for words like `possible_matches`, `pm`, `l1`, `l2p`, or `mask_game`.  Replace historical names with paper-aligned names unless the old word appears only in a compatibility note.
9. Check the output type.  If the phase returns a tuple of more than two conceptual objects, consider creating a named struct.
10. Add or update one doc paragraph explaining why the phase is needed.

Reason: Do not let public exposition suggest the method is only LR. The table is one way to obtain stack effects.

## Task 5: verify `BuildTerminalGrammarFacts`

1. Find the module that implements `BuildTerminalGrammarFacts`.
2. Confirm it consumes `GLRTable + AnalyzedGrammar`.
3. Confirm it produces `TerminalColoring + disallowed follows`.
4. Read the module comment and check that it says: terminal coexistence and legality facts.
5. Search the module for `std::env`.  If found, move the decision to `src/compile/options.rs` unless the module is itself `options.rs`.
6. Search the module for `eprintln!`.  If found, move rendering to `src/compile/profiling.rs` unless the module is explicitly a profile sink.
7. Search the module for `Constraint {`.  Only `FinalizeRuntime` should initialize the runtime constraint.
8. Search the module for words like `possible_matches`, `pm`, `l1`, `l2p`, or `mask_game`.  Replace historical names with paper-aligned names unless the old word appears only in a compatibility note.
9. Check the output type.  If the phase returns a tuple of more than two conceptual objects, consider creating a named struct.
10. Add or update one doc paragraph explaining why the phase is needed.

Reason: These facts help Terminal-DWA construction without changing Terminal-DWA denotation.

## Task 6: verify `BuildTerminalDwa`

1. Find the module that implements `BuildTerminalDwa`.
2. Confirm it consumes `Tokenizer + Vocab + terminal facts`.
3. Confirm it produces `MappedArtifact<DWA>`.
4. Read the module comment and check that it says: complete terminal sequence to lexer-state/token mask.
5. Search the module for `std::env`.  If found, move the decision to `src/compile/options.rs` unless the module is itself `options.rs`.
6. Search the module for `eprintln!`.  If found, move rendering to `src/compile/profiling.rs` unless the module is explicitly a profile sink.
7. Search the module for `Constraint {`.  Only `FinalizeRuntime` should initialize the runtime constraint.
8. Search the module for words like `possible_matches`, `pm`, `l1`, `l2p`, or `mask_game`.  Replace historical names with paper-aligned names unless the old word appears only in a compatibility note.
9. Check the output type.  If the phase returns a tuple of more than two conceptual objects, consider creating a named struct.
10. Add or update one doc paragraph explaining why the phase is needed.

Reason: This is the paper Terminal DWA. Preserve the name and keep complete-vs-partial scan distinction explicit.

## Task 7: verify `BuildScanRelation`

1. Find the module that implements `BuildScanRelation`.
2. Confirm it consumes `Tokenizer + Vocab`.
3. Confirm it produces `ScanRelationComputation`.
4. Read the module comment and check that it says: partial byte-fragment completability.
5. Search the module for `std::env`.  If found, move the decision to `src/compile/options.rs` unless the module is itself `options.rs`.
6. Search the module for `eprintln!`.  If found, move rendering to `src/compile/profiling.rs` unless the module is explicitly a profile sink.
7. Search the module for `Constraint {`.  Only `FinalizeRuntime` should initialize the runtime constraint.
8. Search the module for words like `possible_matches`, `pm`, `l1`, `l2p`, or `mask_game`.  Replace historical names with paper-aligned names unless the old word appears only in a compatibility note.
9. Check the output type.  If the phase returns a tuple of more than two conceptual objects, consider creating a named struct.
10. Add or update one doc paragraph explaining why the phase is needed.

Reason: Never reuse Terminal-DWA quotienting as a correctness proof here.

## Task 8: verify `BuildTemplates`

1. Find the module that implements `BuildTemplates`.
2. Confirm it consumes `GLRTable + AnalyzedGrammar`.
3. Confirm it produces `Templates + TemplateDfasByTerminal`.
4. Read the module comment and check that it says: stack-effect recognizers.
5. Search the module for `std::env`.  If found, move the decision to `src/compile/options.rs` unless the module is itself `options.rs`.
6. Search the module for `eprintln!`.  If found, move rendering to `src/compile/profiling.rs` unless the module is explicitly a profile sink.
7. Search the module for `Constraint {`.  Only `FinalizeRuntime` should initialize the runtime constraint.
8. Search the module for words like `possible_matches`, `pm`, `l1`, `l2p`, or `mask_game`.  Replace historical names with paper-aligned names unless the old word appears only in a compatibility note.
9. Check the output type.  If the phase returns a tuple of more than two conceptual objects, consider creating a named struct.
10. Add or update one doc paragraph explaining why the phase is needed.

Reason: Templates are parser-abstract; avoid over-explaining LR mechanics.

## Task 9: verify `BuildParserDwa`

1. Find the module that implements `BuildParserDwa`.
2. Confirm it consumes `Terminal DWA + Templates + parser facts`.
3. Confirm it produces `Parser DWA`.
4. Read the module comment and check that it says: stack-prefix to token mask automaton.
5. Search the module for `std::env`.  If found, move the decision to `src/compile/options.rs` unless the module is itself `options.rs`.
6. Search the module for `eprintln!`.  If found, move rendering to `src/compile/profiling.rs` unless the module is explicitly a profile sink.
7. Search the module for `Constraint {`.  Only `FinalizeRuntime` should initialize the runtime constraint.
8. Search the module for words like `possible_matches`, `pm`, `l1`, `l2p`, or `mask_game`.  Replace historical names with paper-aligned names unless the old word appears only in a compatibility note.
9. Check the output type.  If the phase returns a tuple of more than two conceptual objects, consider creating a named struct.
10. Add or update one doc paragraph explaining why the phase is needed.

Reason: The Parser DWA is the runtime Mask object over active stack prefixes.

## Task 10: verify `ReconcileArtifact`

1. Find the module that implements `ReconcileArtifact`.
2. Confirm it consumes `Parser DWA + CanMatch + Terminal DWA maps`.
3. Confirm it produces `ReconciledArtifacts`.
4. Read the module comment and check that it says: one shared internal coordinate system.
5. Search the module for `std::env`.  If found, move the decision to `src/compile/options.rs` unless the module is itself `options.rs`.
6. Search the module for `eprintln!`.  If found, move rendering to `src/compile/profiling.rs` unless the module is explicitly a profile sink.
7. Search the module for `Constraint {`.  Only `FinalizeRuntime` should initialize the runtime constraint.
8. Search the module for words like `possible_matches`, `pm`, `l1`, `l2p`, or `mask_game`.  Replace historical names with paper-aligned names unless the old word appears only in a compatibility note.
9. Check the output type.  If the phase returns a tuple of more than two conceptual objects, consider creating a named struct.
10. Add or update one doc paragraph explaining why the phase is needed.

Reason: This phase is a proof that all final weights speak the same ID language.

## Task 11: verify `FinalizeRuntime`

1. Find the module that implements `FinalizeRuntime`.
2. Confirm it consumes `ReconciledArtifacts + Vocab`.
3. Confirm it produces `Constraint`.
4. Read the module comment and check that it says: runtime representation.
5. Search the module for `std::env`.  If found, move the decision to `src/compile/options.rs` unless the module is itself `options.rs`.
6. Search the module for `eprintln!`.  If found, move rendering to `src/compile/profiling.rs` unless the module is explicitly a profile sink.
7. Search the module for `Constraint {`.  Only `FinalizeRuntime` should initialize the runtime constraint.
8. Search the module for words like `possible_matches`, `pm`, `l1`, `l2p`, or `mask_game`.  Replace historical names with paper-aligned names unless the old word appears only in a compatibility note.
9. Check the output type.  If the phase returns a tuple of more than two conceptual objects, consider creating a named struct.
10. Add or update one doc paragraph explaining why the phase is needed.

Reason: Dense caches and sparse caches are derived representations only.
