# glrmask

Fast grammar-constrained decoding for LLMs. Compiles context-free grammars into a
Deterministic Weighted Automaton (DWA) and produces token masks in microseconds.

## Quick Start

```rust
use glrmask::{Constraint, Vocab};

// Build a vocabulary (token_id, bytes).
let vocab = Vocab::new(
    vec![
        (0, b"a".to_vec()),
        (1, b"b".to_vec()),
    ],
    None,
);

// Compile a grammar into a constraint.
let constraint = Constraint::from_ebnf(r#"start ::= "a" "b""#, &vocab).unwrap();

// Create mutable state and step through tokens.
let mut state = constraint.start();
let mask = state.compute_mask(&constraint);
assert!(mask.get(0)); // "a" allowed
assert!(!mask.get(1)); // "b" not yet

state.commit(&constraint, 0).unwrap(); // commit "a"
let mask = state.compute_mask(&constraint);
assert!(mask.get(1)); // now "b" is allowed

state.commit(&constraint, 1).unwrap(); // commit "b"
assert!(state.is_accepting(&constraint));
```

## Grammar Formats

### EBNF

```rust
let c = Constraint::from_ebnf(r#"
    start ::= greeting " " name
    greeting ::= "hello" | "hi"
    name ::= "world" | "rust"
"#, &vocab)?;
```

### Lark

```rust
let c = Constraint::from_lark(r#"
    start: greeting " " name
    greeting: "hello" | "hi"
    name: "world" | "rust"
"#, &vocab)?;
```

### JSON Schema

```rust
let c = Constraint::from_json_schema(r#"{"type": "boolean"}"#, &vocab)?;
```

## Serialization

Compiled constraints can be saved and loaded to avoid recompilation:

```rust
// Save
let bytes = constraint.save()?;
constraint.save_to_file(std::path::Path::new("grammar.bin"))?;

// Load
let c = Constraint::load(&bytes)?;
let c = Constraint::load_from_file(std::path::Path::new("grammar.bin"))?;
```

## Architecture

```
Grammar (EBNF/Lark/JSON Schema)
  → GrammarDef (flat CFG rules)
  → GLR table (states, actions, gotos)
  → NWA (Nested Word Automaton with weights)
  → DWA (Deterministic Weighted Automaton)
  → Constraint (compiled, serializable)
       ↓
  ConstraintState (per-sequence mutable state)
       ↓
  compute_mask() → BitSet  (microsecond-scale)
  commit(token_id)
  is_accepting()
```

### Key Types

| Type | Description |
|------|-------------|
| `Constraint` | Immutable compiled grammar. Thread-safe. |
| `ConstraintState` | Mutable per-sequence state. |
| `Vocab` | Token vocabulary mapping IDs to byte sequences. |
| `BitSet` | Dense bit vector for token masks. |

## Utilities

```rust
use glrmask::runtime::force::{forced_token, is_dead};

let mask = state.compute_mask(&constraint);

// Check if exactly one token is allowed (can skip sampling).
if let Some(token) = forced_token(&mask) {
    state.commit(&constraint, token)?;
}

// Check if no tokens are allowed (dead state).
if is_dead(&mask) {
    // Handle error
}
```

## License

MIT
