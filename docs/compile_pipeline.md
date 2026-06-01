# Compile pipeline

The compile pipeline is now documented as an explicit phase graph.  See `docs/refactor/chunk_03/` for the implementation manual, mathematical contracts, symbol move table, application instructions, and review checklist.

| Order | Phase | Consumes | Produces | Mathematical meaning | Publication-facing rule |
| ---: | --- | --- | --- | --- | --- |
| 0 | ImportNormalize | frontend `GrammarDef` | prepared `GrammarDef` | choose a grammar normal form before any automaton construction | no tokenizer, DWA, parser table, or runtime cache decisions here |
| 1 | BuildTokenizer | prepared grammar terminals | lexer/tokenizer DFA | recognizer for grammar terminals over bytes | grammar object only; no vocabulary quotienting |
| 2 | AnalyzeGrammar | prepared grammar | analyzed parser facts | grammar and stack-effect facts needed by later phases | parser implementation may vary; outputs are facts, not exposition |
| 3 | BuildGlrTable | analyzed grammar | parse table | current implementation witness for stack evolution | should remain behind compile boundary |
| 4 | BuildTerminalGrammarFacts | table + analyzed grammar | terminal coloring, disallowed follows | structural facts about which terminals can coexist in parser states | no vocabulary bytes here except later consumers |
| 5 | BuildTerminalDwa | tokenizer + vocab + terminal facts | Terminal DWA over terminal strings | maps complete terminal sequences to `(lexer-state, token)` masks | name must stay Terminal DWA, not generic DWA or PM artifact |
| 6 | BuildScanRelation | tokenizer + vocab | scan relation / CanMatch | handles byte fragments that may end in a partial terminal | must not reuse Terminal-DWA equivalence as proof |
| 7 | BuildTemplates | parser stack-effect facts | template DFAs | parser-stack effect recognizers used by commit and Parser DWA | not LR-specific in wording |
| 8 | BuildParserDwa | table + templates + Terminal DWA | Parser DWA | maps parser stack prefixes to lexer-state/token masks | depends on stack-effect recognizers, not frontend parsing details |
| 9 | ReconcileArtifact | Terminal DWA, Parser DWA, CanMatch | shared internal coordinate system | proves all masks speak the same internal token/state language | the only place artifact ID spaces merge |
| 10 | FinalizeRuntime | reconciled artifacts + vocab | `Constraint` | package mathematical artifacts into runtime caches | only phase allowed to know `Constraint` field layout |
