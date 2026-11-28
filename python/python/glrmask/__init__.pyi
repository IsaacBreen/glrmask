"""GLRMask is a grammar-constrained decoding library for high-throughput LLM serving. It is optimized for fast mask computation with extremely low tail latency, even for complex grammars.

GLRMask supports general context-free grammars, including ambiguous ones. It accepts JSON Schema, Lark and [GLRM](https://github.com/IsaacBreen/glrmask/blob/main/docs/glrm-format.md).

## Installation

```bash
python -m pip install glrmask
```

## Usage

GLRMask has four core types:

- `Grammar`: a grammar, written as source text or built from an expression.
- `Constraint`: a compiled grammar, ready to use.
- `ConstraintState`: the state of one generated sequence.
- `Module`: a partially compiled grammar ([see below](#composing-grammars)).

In the quickstart, supply the tokenizer's exact token bytes and the model's EOS ID. The model produces logits; the sampling routine must apply the packed mask and permit EOS only when the state is accepting.

## Quickstart

```python
grammar = Grammar.from_json_schema({
    "type": "object",
    "properties": {
        "color": {
            "enum": ["red", "green", "blue"]
        }
    },
    "required": ["color"],
    "additionalProperties": False,
})

# Token ID -> exact bytes from tokenizer
vocab = Vocab(token_bytes)
state = grammar.compile(vocab).start()

output = []
while True:
    logits = model(output)
    mask = state.mask()
    token = sample(
        logits, mask, state.is_accepting()
    )
    if token == EOS:
        break
    if not state.commit_token(token):
        raise RuntimeError("Rejected token")
    output.append(token)
```

## Composing grammars

GLRMask compiles grammars in parts and links them later. A `Module` is a grammar with named externs. Bind each extern to another grammar, a compiled constraint or a token set, then compile. Parts that are already compiled are reused, so linking avoids recompiling them from scratch.

```python
# Token ID -> exact bytes from tokenizer
vocab = Vocab(token_bytes)

weather = Grammar.from_json_schema({
    "type": "object",
    "properties": {
        "city": {"type": "string"}
    },
    "required": ["city"],
    "additionalProperties": False,
}).compile(vocab)

parent = Grammar.from_glrm('''
    extern grammar ARGS
    start call
    nt call = "<weather>" ARGS
              "</weather>"
''')
call = Module(parent)
call.bind("ARGS", weather)
constraint = call.compile(vocab)
```

## Learn more

[Full guide](https://github.com/IsaacBreen/glrmask/blob/main/GUIDE.md) · [API reference](https://glrmask.com/api/python/)
"""
from __future__ import annotations
from collections.abc import Mapping
from typing import Any, ClassVar, Iterable, Literal, NoReturn, final
import numpy as np
from numpy.typing import NDArray

__version__: str
__all__: list[str] = [
    "Vocab",
    "Grammar",
    "Expr",
    "TokenSet",
    "Module",
    "Constraint",
    "ConstraintState",
    "MaskStrategy",
    "Externs",
    "Extern",
    "GLRMaskError",
    "__version__",
]

@final
class Vocab:
    """Each token ID and its bytes."""
    def __init__(self, id_to_bytes: Mapping[int, bytes], /) -> None:
        ...

@final
class Grammar:
    def __new__(cls, _private: NoReturn, /) -> Grammar: ...
    @staticmethod
    def from_glrm(source: str, /) -> Grammar:
        """Parses [GLRM](https://github.com/IsaacBreen/glrmask/blob/v1.0.0-rc.1/docs/glrm-format.md) source."""
        ...
    @staticmethod
    def from_lark(source: str, /) -> Grammar:
        """Parses [Lark](https://github.com/IsaacBreen/glrmask/blob/v1.0.0-rc.1/GUIDE.md#grammar-formats) source."""
        ...
    @staticmethod
    def from_json_schema(schema: str | bool | dict[str, Any], /) -> Grammar:
        """Parses a [JSON Schema](https://github.com/IsaacBreen/glrmask/blob/v1.0.0-rc.1/GUIDE.md#grammar-formats), given as JSON text or a dict."""
        ...
    @staticmethod
    def from_expr(expr: Expr, /) -> Grammar:
        """Builds a grammar from an `Expr`."""
        ...
    @property
    def externs(self) -> Externs:
        ...
    def compile(
        self,
        vocab: Vocab,
        *,
        mask_strategy: MaskStrategy = MaskStrategy.AUTO,
    ) -> Constraint:
        """Compiles the grammar into a constraint."""
        ...
    def copy(self) -> Grammar:
        ...
    def __copy__(self) -> Grammar:
        ...
    def __deepcopy__(self, memo: dict[int, Any]) -> Grammar:
        ...

@final
class Expr:
    """A grammar expression."""
    def __new__(cls, _private: NoReturn, /) -> Expr: ...
    @staticmethod
    def literal(value: str | bytes, /) -> Expr:
        """Matches `value` exactly."""
        ...
    @staticmethod
    def sequence(*parts: Grammar | Module | Constraint | Expr | TokenSet) -> Expr:
        """Matches `parts` in order."""
        ...
    @staticmethod
    def choice(*parts: Grammar | Module | Constraint | Expr | TokenSet) -> Expr:
        """Matches any one of `parts`."""
        ...

@final
class TokenSet:
    """A set of token IDs."""
    def __init__(self, ids: Iterable[int | range], /) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def __eq__(self, other: object, /) -> bool:
        ...

    def __hash__(self) -> int: ...

@final
class Module:
    """A partially compiled grammar."""
    def __init__(self, source: Grammar, /) -> None:
        ...
    @property
    def externs(self) -> Externs:
        ...
    def bind(self, name: str, value: Grammar | Module | Constraint | Expr | TokenSet, /) -> None:
        """Binds an extern by name."""
        ...
    def precompile(
        self,
        vocab: Vocab,
        *,
        mask_strategy: MaskStrategy = MaskStrategy.AUTO,
    ) -> None:
        """Compiles what it can now, even with externs unbound. Further compilation must use the same `vocab`."""
        ...
    def compile(
        self,
        vocab: Vocab,
        *,
        mask_strategy: MaskStrategy = MaskStrategy.AUTO,
    ) -> Constraint:
        """Finishes compiling the module into a constraint."""
        ...
    def save(self) -> bytes:
        ...
    @staticmethod
    def load(data: bytes, /) -> Module:
        """May fail if saved by a different GLRMask version."""
        ...
    def copy(self) -> Module:
        ...
    def __copy__(self) -> Module:
        ...
    def __deepcopy__(self, memo: dict[int, Any]) -> Module:
        ...

@final
class Constraint:
    def __new__(cls, _private: NoReturn, /) -> Constraint: ...
    def __repr__(self) -> str:
        ...
    def start(self) -> ConstraintState:
        """Starts a new sequence."""
        ...
    @property
    def mask_bit_len(self) -> int:
        """The mask's length in bits (max token ID plus one)."""
        ...
    def save(self) -> bytes:
        ...
    @staticmethod
    def load(data: bytes, vocab: Vocab, /) -> Constraint:
        """May fail if saved by a different GLRMask version. Fails if `vocab` differs from the one it was compiled with."""
        ...

@final
class ConstraintState:
    def __new__(cls, _private: NoReturn, /) -> ConstraintState: ...
    def __repr__(self) -> str:
        ...
    def mask(self) -> NDArray[np.uint32]:
        """Packed uint32 words: token i is bit i % 32 of word i // 32."""
        ...
    def fill_mask(self, bitmask: NDArray[np.uint32], /) -> None:
        """Like `mask()`, but writes the words into `bitmask`. May release the GIL."""
        ...
    @property
    def mask_bit_len(self) -> int:
        """The mask's length in bits (max token ID plus one)."""
        ...
    def commit_token(self, token_id: int, /) -> bool:
        """Advances the state by the token. If it isn't allowed, returns `False` and terminates the state."""
        ...
    def commit_bytes(self, data: bytes, /) -> bool:
        """Advances the state by the bytes. If they aren't allowed, returns `False` and terminates the state."""
        ...
    def forced_bytes(self) -> bytes:
        """The bytes that must come next."""
        ...
    def is_accepting(self) -> bool:
        """Whether the output so far could end here."""
        ...
    def is_terminated(self) -> bool:
        """Whether nothing more can be committed."""
        ...
    def copy(self) -> ConstraintState:
        ...
    def __copy__(self) -> ConstraintState:
        ...
    def __deepcopy__(self, memo: dict[int, Any]) -> ConstraintState:
        ...

@final
class MaskStrategy:
    """Controls how the constraint is compiled."""
    def __new__(cls, _private: NoReturn, /) -> MaskStrategy: ...
    AUTO: ClassVar[MaskStrategy]
    """The default; currently `FAST_MASKS`."""
    FAST_COMPILE: ClassVar[MaskStrategy]
    """Faster compilation, slower masks."""
    FAST_MASKS: ClassVar[MaskStrategy]
    """Faster masks, slower compilation."""

    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

@final
class Externs(Mapping[str, "Extern"]):
    """A read-only mapping of extern names to `Extern`."""
    def __new__(cls, _private: NoReturn, /) -> Externs: ...
    @property
    def unbound(self) -> Externs:
        ...
    def __repr__(self) -> str:
        ...
    def __eq__(self, other: object, /) -> bool:
        ...

@final
class Extern:
    """A named placeholder in a grammar."""
    def __new__(cls, _private: NoReturn, /) -> Extern: ...
    @property
    def name(self) -> str:
        ...
    @property
    def kind(self) -> Literal["token", "grammar"]:
        ...
    def __repr__(self) -> str:
        ...
    def __eq__(self, other: object, /) -> bool:
        ...

    def __hash__(self) -> int: ...

class GLRMaskError(Exception):
    kind: Literal["syntax", "invalid_input", "compilation", "resource_limit", "strategy_unsupported", "serialization", "vocab_mismatch", "buffer_too_small", "internal"]
    """One of `"syntax"`, `"invalid_input"`, `"compilation"`, `"resource_limit"`, `"strategy_unsupported"`, `"serialization"`, `"vocab_mismatch"`, `"buffer_too_small"`, `"internal"`."""
