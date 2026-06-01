# Chunk 03 phase-by-phase deep dive

This document expands every phase into input/output contracts, ownership rules, review questions, and failure modes.  It is intentionally repetitive because the repetition makes the intended architecture unambiguous.

## ImportNormalize

### Input and output

- **Input:** `frontend grammar`.
- **Output:** `prepared GrammarDef`.
- **Mathematical meaning:** language-preserving grammar normal form.
- **Non-negotiable rule:** Do not inspect token bytes here. Do not decide artifact coordinate systems here. This is a source-shape phase.

### Why this phase exists

This phase exists because the compile pipeline needs a boundary at exactly this concept.  Without the boundary, a maintainer has to infer the object from local variables and incidental call order.  For a publication crate, that is backwards: the source tree should make the proof structure obvious before the reader studies optimization details.

### What belongs here

Code belongs in this phase if it directly constructs the output above, checks a precondition required to construct that output, or records a profile number whose interpretation is tied to that output.  Helper functions are acceptable only if they are not reusable concepts in their own right.  Once a helper acquires an independent mathematical name, it should move to its own module.

### What does not belong here

Runtime cache layout does not belong here unless this is `FinalizeRuntime`.  Environment parsing does not belong here.  Profile string formatting does not belong here.  Frontend source-language quirks do not belong here after normalization.  A phase should not know about future phases except through the type of the value it returns.

### Invariant to state in comments

A useful comment near this phase should answer: "What is the denotation of the value produced here?"  It should not merely say what algorithm currently computes it.  The algorithm can change while the denotation remains fixed.

### Failure mode if this boundary is violated

If this boundary is violated, later cleanup becomes impossible without re-reading the whole compiler.  In particular, performance hacks can begin to look like semantic requirements.  The publication reader should be able to distinguish semantic facts, quotienting choices, and runtime layout choices.

### Concrete review questions

1. Does the phase read any global configuration directly?
2. Does the phase print directly?
3. Does the phase mutate a runtime cache before finalization?
4. Does the phase use a name from the paper or a historical implementation nickname?
5. Could a different parser construction still satisfy this phase's input/output contract?
6. Is any local variable carrying a concept that deserves a named output type?
7. Are profile counters measuring this phase or an unrelated downstream effect?
8. Are panics/preconditions written in terms of phase obligations?
9. Does the module doc explain denotation, not just implementation mechanics?
10. Would a new contributor know where to add related code?

## BuildTokenizer

### Input and output

- **Input:** `prepared GrammarDef terminals`.
- **Output:** `Tokenizer`.
- **Mathematical meaning:** byte-level recognition of grammar terminals.
- **Non-negotiable rule:** The tokenizer is grammar-only. Vocabulary appears later when token bytes are classified.

### Why this phase exists

This phase exists because the compile pipeline needs a boundary at exactly this concept.  Without the boundary, a maintainer has to infer the object from local variables and incidental call order.  For a publication crate, that is backwards: the source tree should make the proof structure obvious before the reader studies optimization details.

### What belongs here

Code belongs in this phase if it directly constructs the output above, checks a precondition required to construct that output, or records a profile number whose interpretation is tied to that output.  Helper functions are acceptable only if they are not reusable concepts in their own right.  Once a helper acquires an independent mathematical name, it should move to its own module.

### What does not belong here

Runtime cache layout does not belong here unless this is `FinalizeRuntime`.  Environment parsing does not belong here.  Profile string formatting does not belong here.  Frontend source-language quirks do not belong here after normalization.  A phase should not know about future phases except through the type of the value it returns.

### Invariant to state in comments

A useful comment near this phase should answer: "What is the denotation of the value produced here?"  It should not merely say what algorithm currently computes it.  The algorithm can change while the denotation remains fixed.

### Failure mode if this boundary is violated

If this boundary is violated, later cleanup becomes impossible without re-reading the whole compiler.  In particular, performance hacks can begin to look like semantic requirements.  The publication reader should be able to distinguish semantic facts, quotienting choices, and runtime layout choices.

### Concrete review questions

1. Does the phase read any global configuration directly?
2. Does the phase print directly?
3. Does the phase mutate a runtime cache before finalization?
4. Does the phase use a name from the paper or a historical implementation nickname?
5. Could a different parser construction still satisfy this phase's input/output contract?
6. Is any local variable carrying a concept that deserves a named output type?
7. Are profile counters measuring this phase or an unrelated downstream effect?
8. Are panics/preconditions written in terms of phase obligations?
9. Does the module doc explain denotation, not just implementation mechanics?
10. Would a new contributor know where to add related code?

## AnalyzeGrammar

### Input and output

- **Input:** `prepared GrammarDef productions`.
- **Output:** `AnalyzedGrammar`.
- **Mathematical meaning:** parser-side semantic facts.
- **Non-negotiable rule:** Keep this separate from specific runtime masking decisions. The parser is a provider of stack-effect facts.

### Why this phase exists

This phase exists because the compile pipeline needs a boundary at exactly this concept.  Without the boundary, a maintainer has to infer the object from local variables and incidental call order.  For a publication crate, that is backwards: the source tree should make the proof structure obvious before the reader studies optimization details.

### What belongs here

Code belongs in this phase if it directly constructs the output above, checks a precondition required to construct that output, or records a profile number whose interpretation is tied to that output.  Helper functions are acceptable only if they are not reusable concepts in their own right.  Once a helper acquires an independent mathematical name, it should move to its own module.

### What does not belong here

Runtime cache layout does not belong here unless this is `FinalizeRuntime`.  Environment parsing does not belong here.  Profile string formatting does not belong here.  Frontend source-language quirks do not belong here after normalization.  A phase should not know about future phases except through the type of the value it returns.

### Invariant to state in comments

A useful comment near this phase should answer: "What is the denotation of the value produced here?"  It should not merely say what algorithm currently computes it.  The algorithm can change while the denotation remains fixed.

### Failure mode if this boundary is violated

If this boundary is violated, later cleanup becomes impossible without re-reading the whole compiler.  In particular, performance hacks can begin to look like semantic requirements.  The publication reader should be able to distinguish semantic facts, quotienting choices, and runtime layout choices.

### Concrete review questions

1. Does the phase read any global configuration directly?
2. Does the phase print directly?
3. Does the phase mutate a runtime cache before finalization?
4. Does the phase use a name from the paper or a historical implementation nickname?
5. Could a different parser construction still satisfy this phase's input/output contract?
6. Is any local variable carrying a concept that deserves a named output type?
7. Are profile counters measuring this phase or an unrelated downstream effect?
8. Are panics/preconditions written in terms of phase obligations?
9. Does the module doc explain denotation, not just implementation mechanics?
10. Would a new contributor know where to add related code?

## BuildGlrTable

### Input and output

- **Input:** `AnalyzedGrammar`.
- **Output:** `GLRTable`.
- **Mathematical meaning:** implementation witness for stack evolution.
- **Non-negotiable rule:** Do not let public exposition suggest the method is only LR. The table is one way to obtain stack effects.

### Why this phase exists

This phase exists because the compile pipeline needs a boundary at exactly this concept.  Without the boundary, a maintainer has to infer the object from local variables and incidental call order.  For a publication crate, that is backwards: the source tree should make the proof structure obvious before the reader studies optimization details.

### What belongs here

Code belongs in this phase if it directly constructs the output above, checks a precondition required to construct that output, or records a profile number whose interpretation is tied to that output.  Helper functions are acceptable only if they are not reusable concepts in their own right.  Once a helper acquires an independent mathematical name, it should move to its own module.

### What does not belong here

Runtime cache layout does not belong here unless this is `FinalizeRuntime`.  Environment parsing does not belong here.  Profile string formatting does not belong here.  Frontend source-language quirks do not belong here after normalization.  A phase should not know about future phases except through the type of the value it returns.

### Invariant to state in comments

A useful comment near this phase should answer: "What is the denotation of the value produced here?"  It should not merely say what algorithm currently computes it.  The algorithm can change while the denotation remains fixed.

### Failure mode if this boundary is violated

If this boundary is violated, later cleanup becomes impossible without re-reading the whole compiler.  In particular, performance hacks can begin to look like semantic requirements.  The publication reader should be able to distinguish semantic facts, quotienting choices, and runtime layout choices.

### Concrete review questions

1. Does the phase read any global configuration directly?
2. Does the phase print directly?
3. Does the phase mutate a runtime cache before finalization?
4. Does the phase use a name from the paper or a historical implementation nickname?
5. Could a different parser construction still satisfy this phase's input/output contract?
6. Is any local variable carrying a concept that deserves a named output type?
7. Are profile counters measuring this phase or an unrelated downstream effect?
8. Are panics/preconditions written in terms of phase obligations?
9. Does the module doc explain denotation, not just implementation mechanics?
10. Would a new contributor know where to add related code?

## BuildTerminalGrammarFacts

### Input and output

- **Input:** `GLRTable + AnalyzedGrammar`.
- **Output:** `TerminalColoring + disallowed follows`.
- **Mathematical meaning:** terminal coexistence and legality facts.
- **Non-negotiable rule:** These facts help Terminal-DWA construction without changing Terminal-DWA denotation.

### Why this phase exists

This phase exists because the compile pipeline needs a boundary at exactly this concept.  Without the boundary, a maintainer has to infer the object from local variables and incidental call order.  For a publication crate, that is backwards: the source tree should make the proof structure obvious before the reader studies optimization details.

### What belongs here

Code belongs in this phase if it directly constructs the output above, checks a precondition required to construct that output, or records a profile number whose interpretation is tied to that output.  Helper functions are acceptable only if they are not reusable concepts in their own right.  Once a helper acquires an independent mathematical name, it should move to its own module.

### What does not belong here

Runtime cache layout does not belong here unless this is `FinalizeRuntime`.  Environment parsing does not belong here.  Profile string formatting does not belong here.  Frontend source-language quirks do not belong here after normalization.  A phase should not know about future phases except through the type of the value it returns.

### Invariant to state in comments

A useful comment near this phase should answer: "What is the denotation of the value produced here?"  It should not merely say what algorithm currently computes it.  The algorithm can change while the denotation remains fixed.

### Failure mode if this boundary is violated

If this boundary is violated, later cleanup becomes impossible without re-reading the whole compiler.  In particular, performance hacks can begin to look like semantic requirements.  The publication reader should be able to distinguish semantic facts, quotienting choices, and runtime layout choices.

### Concrete review questions

1. Does the phase read any global configuration directly?
2. Does the phase print directly?
3. Does the phase mutate a runtime cache before finalization?
4. Does the phase use a name from the paper or a historical implementation nickname?
5. Could a different parser construction still satisfy this phase's input/output contract?
6. Is any local variable carrying a concept that deserves a named output type?
7. Are profile counters measuring this phase or an unrelated downstream effect?
8. Are panics/preconditions written in terms of phase obligations?
9. Does the module doc explain denotation, not just implementation mechanics?
10. Would a new contributor know where to add related code?

## BuildTerminalDwa

### Input and output

- **Input:** `Tokenizer + Vocab + terminal facts`.
- **Output:** `MappedArtifact<DWA>`.
- **Mathematical meaning:** complete terminal sequence to lexer-state/token mask.
- **Non-negotiable rule:** This is the paper Terminal DWA. Preserve the name and keep complete-vs-partial scan distinction explicit.

### Why this phase exists

This phase exists because the compile pipeline needs a boundary at exactly this concept.  Without the boundary, a maintainer has to infer the object from local variables and incidental call order.  For a publication crate, that is backwards: the source tree should make the proof structure obvious before the reader studies optimization details.

### What belongs here

Code belongs in this phase if it directly constructs the output above, checks a precondition required to construct that output, or records a profile number whose interpretation is tied to that output.  Helper functions are acceptable only if they are not reusable concepts in their own right.  Once a helper acquires an independent mathematical name, it should move to its own module.

### What does not belong here

Runtime cache layout does not belong here unless this is `FinalizeRuntime`.  Environment parsing does not belong here.  Profile string formatting does not belong here.  Frontend source-language quirks do not belong here after normalization.  A phase should not know about future phases except through the type of the value it returns.

### Invariant to state in comments

A useful comment near this phase should answer: "What is the denotation of the value produced here?"  It should not merely say what algorithm currently computes it.  The algorithm can change while the denotation remains fixed.

### Failure mode if this boundary is violated

If this boundary is violated, later cleanup becomes impossible without re-reading the whole compiler.  In particular, performance hacks can begin to look like semantic requirements.  The publication reader should be able to distinguish semantic facts, quotienting choices, and runtime layout choices.

### Concrete review questions

1. Does the phase read any global configuration directly?
2. Does the phase print directly?
3. Does the phase mutate a runtime cache before finalization?
4. Does the phase use a name from the paper or a historical implementation nickname?
5. Could a different parser construction still satisfy this phase's input/output contract?
6. Is any local variable carrying a concept that deserves a named output type?
7. Are profile counters measuring this phase or an unrelated downstream effect?
8. Are panics/preconditions written in terms of phase obligations?
9. Does the module doc explain denotation, not just implementation mechanics?
10. Would a new contributor know where to add related code?

## BuildScanRelation

### Input and output

- **Input:** `Tokenizer + Vocab`.
- **Output:** `ScanRelationComputation`.
- **Mathematical meaning:** partial byte-fragment completability.
- **Non-negotiable rule:** Never reuse Terminal-DWA quotienting as a correctness proof here.

### Why this phase exists

This phase exists because the compile pipeline needs a boundary at exactly this concept.  Without the boundary, a maintainer has to infer the object from local variables and incidental call order.  For a publication crate, that is backwards: the source tree should make the proof structure obvious before the reader studies optimization details.

### What belongs here

Code belongs in this phase if it directly constructs the output above, checks a precondition required to construct that output, or records a profile number whose interpretation is tied to that output.  Helper functions are acceptable only if they are not reusable concepts in their own right.  Once a helper acquires an independent mathematical name, it should move to its own module.

### What does not belong here

Runtime cache layout does not belong here unless this is `FinalizeRuntime`.  Environment parsing does not belong here.  Profile string formatting does not belong here.  Frontend source-language quirks do not belong here after normalization.  A phase should not know about future phases except through the type of the value it returns.

### Invariant to state in comments

A useful comment near this phase should answer: "What is the denotation of the value produced here?"  It should not merely say what algorithm currently computes it.  The algorithm can change while the denotation remains fixed.

### Failure mode if this boundary is violated

If this boundary is violated, later cleanup becomes impossible without re-reading the whole compiler.  In particular, performance hacks can begin to look like semantic requirements.  The publication reader should be able to distinguish semantic facts, quotienting choices, and runtime layout choices.

### Concrete review questions

1. Does the phase read any global configuration directly?
2. Does the phase print directly?
3. Does the phase mutate a runtime cache before finalization?
4. Does the phase use a name from the paper or a historical implementation nickname?
5. Could a different parser construction still satisfy this phase's input/output contract?
6. Is any local variable carrying a concept that deserves a named output type?
7. Are profile counters measuring this phase or an unrelated downstream effect?
8. Are panics/preconditions written in terms of phase obligations?
9. Does the module doc explain denotation, not just implementation mechanics?
10. Would a new contributor know where to add related code?

## BuildTemplates

### Input and output

- **Input:** `GLRTable + AnalyzedGrammar`.
- **Output:** `Templates + TemplateDfasByTerminal`.
- **Mathematical meaning:** stack-effect recognizers.
- **Non-negotiable rule:** Templates are parser-abstract; avoid over-explaining LR mechanics.

### Why this phase exists

This phase exists because the compile pipeline needs a boundary at exactly this concept.  Without the boundary, a maintainer has to infer the object from local variables and incidental call order.  For a publication crate, that is backwards: the source tree should make the proof structure obvious before the reader studies optimization details.

### What belongs here

Code belongs in this phase if it directly constructs the output above, checks a precondition required to construct that output, or records a profile number whose interpretation is tied to that output.  Helper functions are acceptable only if they are not reusable concepts in their own right.  Once a helper acquires an independent mathematical name, it should move to its own module.

### What does not belong here

Runtime cache layout does not belong here unless this is `FinalizeRuntime`.  Environment parsing does not belong here.  Profile string formatting does not belong here.  Frontend source-language quirks do not belong here after normalization.  A phase should not know about future phases except through the type of the value it returns.

### Invariant to state in comments

A useful comment near this phase should answer: "What is the denotation of the value produced here?"  It should not merely say what algorithm currently computes it.  The algorithm can change while the denotation remains fixed.

### Failure mode if this boundary is violated

If this boundary is violated, later cleanup becomes impossible without re-reading the whole compiler.  In particular, performance hacks can begin to look like semantic requirements.  The publication reader should be able to distinguish semantic facts, quotienting choices, and runtime layout choices.

### Concrete review questions

1. Does the phase read any global configuration directly?
2. Does the phase print directly?
3. Does the phase mutate a runtime cache before finalization?
4. Does the phase use a name from the paper or a historical implementation nickname?
5. Could a different parser construction still satisfy this phase's input/output contract?
6. Is any local variable carrying a concept that deserves a named output type?
7. Are profile counters measuring this phase or an unrelated downstream effect?
8. Are panics/preconditions written in terms of phase obligations?
9. Does the module doc explain denotation, not just implementation mechanics?
10. Would a new contributor know where to add related code?

## BuildParserDwa

### Input and output

- **Input:** `Terminal DWA + Templates + parser facts`.
- **Output:** `Parser DWA`.
- **Mathematical meaning:** stack-prefix to token mask automaton.
- **Non-negotiable rule:** The Parser DWA is the runtime Mask object over active stack prefixes.

### Why this phase exists

This phase exists because the compile pipeline needs a boundary at exactly this concept.  Without the boundary, a maintainer has to infer the object from local variables and incidental call order.  For a publication crate, that is backwards: the source tree should make the proof structure obvious before the reader studies optimization details.

### What belongs here

Code belongs in this phase if it directly constructs the output above, checks a precondition required to construct that output, or records a profile number whose interpretation is tied to that output.  Helper functions are acceptable only if they are not reusable concepts in their own right.  Once a helper acquires an independent mathematical name, it should move to its own module.

### What does not belong here

Runtime cache layout does not belong here unless this is `FinalizeRuntime`.  Environment parsing does not belong here.  Profile string formatting does not belong here.  Frontend source-language quirks do not belong here after normalization.  A phase should not know about future phases except through the type of the value it returns.

### Invariant to state in comments

A useful comment near this phase should answer: "What is the denotation of the value produced here?"  It should not merely say what algorithm currently computes it.  The algorithm can change while the denotation remains fixed.

### Failure mode if this boundary is violated

If this boundary is violated, later cleanup becomes impossible without re-reading the whole compiler.  In particular, performance hacks can begin to look like semantic requirements.  The publication reader should be able to distinguish semantic facts, quotienting choices, and runtime layout choices.

### Concrete review questions

1. Does the phase read any global configuration directly?
2. Does the phase print directly?
3. Does the phase mutate a runtime cache before finalization?
4. Does the phase use a name from the paper or a historical implementation nickname?
5. Could a different parser construction still satisfy this phase's input/output contract?
6. Is any local variable carrying a concept that deserves a named output type?
7. Are profile counters measuring this phase or an unrelated downstream effect?
8. Are panics/preconditions written in terms of phase obligations?
9. Does the module doc explain denotation, not just implementation mechanics?
10. Would a new contributor know where to add related code?

## ReconcileArtifact

### Input and output

- **Input:** `Parser DWA + CanMatch + Terminal DWA maps`.
- **Output:** `ReconciledArtifacts`.
- **Mathematical meaning:** one shared internal coordinate system.
- **Non-negotiable rule:** This phase is a proof that all final weights speak the same ID language.

### Why this phase exists

This phase exists because the compile pipeline needs a boundary at exactly this concept.  Without the boundary, a maintainer has to infer the object from local variables and incidental call order.  For a publication crate, that is backwards: the source tree should make the proof structure obvious before the reader studies optimization details.

### What belongs here

Code belongs in this phase if it directly constructs the output above, checks a precondition required to construct that output, or records a profile number whose interpretation is tied to that output.  Helper functions are acceptable only if they are not reusable concepts in their own right.  Once a helper acquires an independent mathematical name, it should move to its own module.

### What does not belong here

Runtime cache layout does not belong here unless this is `FinalizeRuntime`.  Environment parsing does not belong here.  Profile string formatting does not belong here.  Frontend source-language quirks do not belong here after normalization.  A phase should not know about future phases except through the type of the value it returns.

### Invariant to state in comments

A useful comment near this phase should answer: "What is the denotation of the value produced here?"  It should not merely say what algorithm currently computes it.  The algorithm can change while the denotation remains fixed.

### Failure mode if this boundary is violated

If this boundary is violated, later cleanup becomes impossible without re-reading the whole compiler.  In particular, performance hacks can begin to look like semantic requirements.  The publication reader should be able to distinguish semantic facts, quotienting choices, and runtime layout choices.

### Concrete review questions

1. Does the phase read any global configuration directly?
2. Does the phase print directly?
3. Does the phase mutate a runtime cache before finalization?
4. Does the phase use a name from the paper or a historical implementation nickname?
5. Could a different parser construction still satisfy this phase's input/output contract?
6. Is any local variable carrying a concept that deserves a named output type?
7. Are profile counters measuring this phase or an unrelated downstream effect?
8. Are panics/preconditions written in terms of phase obligations?
9. Does the module doc explain denotation, not just implementation mechanics?
10. Would a new contributor know where to add related code?

## FinalizeRuntime

### Input and output

- **Input:** `ReconciledArtifacts + Vocab`.
- **Output:** `Constraint`.
- **Mathematical meaning:** runtime representation.
- **Non-negotiable rule:** Dense caches and sparse caches are derived representations only.

### Why this phase exists

This phase exists because the compile pipeline needs a boundary at exactly this concept.  Without the boundary, a maintainer has to infer the object from local variables and incidental call order.  For a publication crate, that is backwards: the source tree should make the proof structure obvious before the reader studies optimization details.

### What belongs here

Code belongs in this phase if it directly constructs the output above, checks a precondition required to construct that output, or records a profile number whose interpretation is tied to that output.  Helper functions are acceptable only if they are not reusable concepts in their own right.  Once a helper acquires an independent mathematical name, it should move to its own module.

### What does not belong here

Runtime cache layout does not belong here unless this is `FinalizeRuntime`.  Environment parsing does not belong here.  Profile string formatting does not belong here.  Frontend source-language quirks do not belong here after normalization.  A phase should not know about future phases except through the type of the value it returns.

### Invariant to state in comments

A useful comment near this phase should answer: "What is the denotation of the value produced here?"  It should not merely say what algorithm currently computes it.  The algorithm can change while the denotation remains fixed.

### Failure mode if this boundary is violated

If this boundary is violated, later cleanup becomes impossible without re-reading the whole compiler.  In particular, performance hacks can begin to look like semantic requirements.  The publication reader should be able to distinguish semantic facts, quotienting choices, and runtime layout choices.

### Concrete review questions

1. Does the phase read any global configuration directly?
2. Does the phase print directly?
3. Does the phase mutate a runtime cache before finalization?
4. Does the phase use a name from the paper or a historical implementation nickname?
5. Could a different parser construction still satisfy this phase's input/output contract?
6. Is any local variable carrying a concept that deserves a named output type?
7. Are profile counters measuring this phase or an unrelated downstream effect?
8. Are panics/preconditions written in terms of phase obligations?
9. Does the module doc explain denotation, not just implementation mechanics?
10. Would a new contributor know where to add related code?
