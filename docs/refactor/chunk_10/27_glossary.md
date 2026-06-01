# Chunk 10 glossary

## artifact

The compiled object that preserves the accepted language and can be serialized.

## runtime cache

A derived data structure used to make Mask/Commit fast; should be rebuildable.

## original token id

The model/tokenizer vocabulary id visible to users.

## internal token id

The final compact token equivalence class used by Parser-DWA and CanMatch weights.

## original tokenizer state

A state in the tokenizer DFA before final quotienting.

## internal tokenizer-state id

The quotient id used in final runtime weights.

## Parser DWA

Deterministic weighted automaton over parser-stack symbols whose outputs are masks over lexer-state/token pairs.

## CanMatch

Relation indicating which terminals/tokens can complete lexer scans from states.

## seed mask

Initial dense mask used to seed Mask traversal from a lexer state.

## materialization

Conversion from internal-token dense/sparse sets to public output masks over original tokens.

## finalization

Rebuild of all derived runtime caches after compile or load.

## serialization envelope

Versioned wrapper around the serialized compiled artifact.

