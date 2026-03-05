# Changelog

## 0.1.0 — Initial Release

### Features
- **EBNF, Lark, and JSON Schema** grammar frontends
- **GLR parser** for ambiguous grammar support
- **DWA-based mask computation** in microseconds
- **Serialization**: `save()`/`load()` via bincode
- **Force detection**: `forced_token()` and `is_dead()` utilities
- 196 tests (179 unit + 17 integration)

### Architecture
- `ds/`: Core data structures (RangeSet, U8Set, BitSet)
- `automata/`: DFA, NFA, regex, weighted automata (NWA, DWA)
- `compiler/`: Grammar → GLR table → NWA → DWA pipeline
- `frontend/`: EBNF, Lark, JSON Schema parsers
- `runtime/`: Constraint state, mask computation, force detection
