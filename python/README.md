# Shingleback Python bindings (`glrmask`)

The Python bindings compile grammars into token constraints and expose incremental mask and commit operations.

## Install from a source checkout

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install ./python
```

Building from source requires a Rust toolchain and the platform linker/build tools (for example, Xcode Command Line Tools on macOS). `pip` installs Maturin in an isolated build environment and installs the Python runtime dependencies.

## Minimal example

```python
import glrmask

vocab = glrmask.Vocab.from_dict({
    b"hello": 0,
    b" ": 1,
    b"world": 2,
})
constraint = glrmask.Constraint.from_ebnf(
    'start ::= "hello" " " "world"',
    vocab,
)
state = constraint.start()

assert state.mask().tolist() == [True, False, False]
state.commit_token(0)
assert state.mask().tolist() == [False, True, False]
state.commit_token(1)
assert state.mask().tolist() == [False, False, True]
state.commit_token(2)
assert state.is_finished()
```
