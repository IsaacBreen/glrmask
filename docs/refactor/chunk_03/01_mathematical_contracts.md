# Chunk 03 mathematical contracts and invariants

## A. Compile graph as proof decomposition

A clean publication implementation should make the proof obligations visible.  The compile graph can be read as a sequence of lemmas.

### Lemma 1: grammar normalization preserves the generated language

Input: frontend grammar.  Output: prepared `GrammarDef`.  Obligation: later automata should not need to know whether the source came from EBNF, Lark, JSON Schema, or GLRM.  Frontend quirks must be discharged before the compile graph starts.

### Lemma 2: tokenizer construction creates the byte-level terminal recognizer

Input: prepared grammar terminals.  Output: tokenizer DFA.  Obligation: the tokenizer recognizes the grammar terminal language.  It does not know about model tokens.  This keeps lexer semantics separate from vocabulary quotienting.

### Lemma 3: parser facts expose stack effects without exposing the whole parser as the paper object

Input: prepared grammar.  Output: analyzed grammar, GLR table, terminal colors, disallowed follows.  Obligation: the current implementation can use GLR machinery, but the paper-level object downstream is not “an LR parser”; it is a source of stack-effect recognizers and legal terminal-transition facts.

### Lemma 4: Terminal DWA denotes complete-terminal token scans

Input: tokenizer, vocabulary, parser-derived terminal facts.  Output: Terminal DWA.  Obligation: evaluating the Terminal DWA on a terminal string returns the mask of lexer-state/token pairs that can produce that terminal string as a sequence of completed terminals.

### Lemma 5: scan relation denotes boundary-sensitive partial scans

Input: tokenizer, vocabulary.  Output: CanMatch artifact.  Obligation: when scanning a byte fragment leaves the lexer in a non-boundary state, the parser must be checked against terminals that can still complete from that partial state.  This is not equivalent to a complete-terminal scan language.

### Lemma 6: template DFAs denote parser stack effects

Input: parser stack-effect facts.  Output: template recognizers.  Obligation: these recognizers encode the parser-side effect needed by both Parser DWA construction and Commit.  Their conceptual contract is about stack effects; it should not be described as a special case of an LR algorithm.

### Lemma 7: Parser DWA denotes stack-prefix acceptance for token masks

Input: table, analyzed grammar, Terminal DWA, templates, internal IDs.  Output: Parser DWA.  Obligation: walking a stack prefix through the Parser DWA returns the mask of lexer-state/token pairs admitted by that parser stack prefix.

### Lemma 8: reconciliation gives one internal coordinate system

Input: Terminal DWA, Parser DWA, CanMatch.  Output: shared internal IDs.  Obligation: after this phase, all final runtime weights use the same internal tokenizer-state IDs and internal token IDs.  Before this phase, equality of internal IDs across artifacts is not meaningful.

### Lemma 9: finalization is representation, not semantics

Input: reconciled artifacts.  Output: `Constraint`.  Obligation: runtime caches preserve the semantics of the artifacts.  Dense masks, sparse masks, byte groups, word groups, and heavy-token caches are implementation accelerators.

## B. Boundary rules

### B.1 Pipeline modules must not read environment variables

The pipeline should be deterministic as a function of its inputs and compile decisions.  Today decisions are still environment-backed, but the environment reads are centralized in `compile::options` and `compile::profiling`.  This is an intermediate step toward true explicit options.

### B.2 Pipeline modules must not print ad-hoc profile strings

Profile rendering belongs in `compile::profiling`.  Pipeline phases update a report; they do not decide how reporting is displayed.

### B.3 Runtime layout knowledge belongs only in finalization

If a compile phase before `finalize.rs` refers to fields like `weight_token_dense_masks`, `seed_state_dense`, `internal_token_buf_offsets`, or `final_mask_mapping`, that phase boundary has failed.

### B.4 Terminal DWA equivalence must not be used as CanMatch equivalence

This is the central correctness warning for this part of the system.  Complete terminal-string equivalence is not partial-scan equivalence.  Two token groups can behave identically for completed terminal sequences while having different possible completions at a fragment boundary.

### B.5 Parser implementation is not paper terminology

The implementation currently uses GLR table machinery.  The paper language should emphasize stack-effect recognizers, Parser DWA, and active stacks.  Code documentation should not make the algorithm sound tied to LR as a concept.  GLR is a current construction strategy, not the mathematical interface.

## C. Coordinate-system invariant

Let `I_T` be the internal coordinate system of the Terminal DWA before reconciliation.  Let `I_C` be the coordinate system of CanMatch before reconciliation.  Let `I_P` be the coordinate system of the Parser DWA if it is built before final Parser/CanMatch reconciliation.  The final runtime constraint must use one coordinate system `I` such that:

- every Parser-DWA weight is interpreted in `I`,
- every CanMatch weight is interpreted in `I`,
- `original_token_to_internal` maps original vocab token IDs into `I`,
- `internal_token_to_tokens` maps each internal token ID in `I` back to exactly the original token IDs it denotes,
- `state_to_internal_tsid` maps tokenizer states into the state side of `I`,
- `internal_tsid_to_states` reverses that state map.

The modes in `DwaCanMatchMode` are implementation strategies for constructing `I`; they are not different semantics.

## D. Future compile report shape

The current `CompilePhaseProfile` remains field-compatible with benchmark scripts.  The eventual publication-quality report should be richer:

```text
CompileReport {
    phases: Vec<PhaseRecord>,
    artifact_sizes: ArtifactSizeReport,
    options: ResolvedCompileOptions,
    warnings: Vec<CompileWarning>,
}
```

Each `PhaseRecord` should contain:

- `phase: CompilePhase`,
- `started_at` or at least `elapsed_ms`,
- input-size summary,
- output-size summary,
- optional implementation notes.

This chunk introduces `CompilePhase` and explicit phase functions so that richer report can be added without reverse-engineering the pipeline later.
