# GLRMask Python bindings (`glrmask`)

GLRMask provides grammar-constrained token masking for context-free languages. The PyPI distribution and Python import are both `glrmask`. The bindings compile grammars together with a token vocabulary and expose incremental mask and commit operations.

## Install

From PyPI:

```bash
python -m pip install glrmask==0.1.0
```

For a source build, from the repository root:


```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install ./python
```

Building from source requires a Rust toolchain and the platform linker/build tools (for example, Xcode Command Line Tools on macOS). `pip` installs Maturin in an isolated build environment and installs the Python runtime dependency (`numpy`).

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

The mask is tokenization-complete for the compiled byte language: a vocabulary token is admitted when appending its bytes leaves at least one valid completion, even when the token crosses grammar-terminal boundaries.
