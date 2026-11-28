"""GLRMask is a grammar-constrained decoding library for high-throughput LLM serving. It is optimized for fast mask computation with extremely low tail latency, even for complex grammars.

GLRMask supports general context-free grammars, including ambiguous ones. It accepts JSON Schema, Lark and [GLRM](https://github.com/IsaacBreen/glrmask/blob/main/docs/glrm-format.md).

## Installation

```bash
python -m pip install glrmask
```

For the release candidate: `pip install --pre glrmask` or `pip install glrmask==1.0.0rc1`.

## Usage

GLRMask has four core types:

- `Grammar`
- `Module`: a partially compiled grammar.
- `Constraint`: a compiled grammar, ready to use.
- `ConstraintState`: the state of one generated sequence.

## Quickstart

```python
import numpy as np
import glrmask

grammar = glrmask.Grammar.from_json_schema({
    "type": "object",
    "properties": {"color": {"enum": ["red", "green", "blue"]}},
    "required": ["color"],
    "additionalProperties": False,
})

# One token per byte, plus EOS. With a real model, build this from its tokenizer.
vocab = glrmask.Vocab({i: bytes([i]) for i in range(256)})
EOS = 256

state = grammar.compile(vocab).start()
rng = np.random.default_rng(0)
output = []
while True:
    logits = rng.standard_normal(257)  # stand-in for your model
    allowed = np.zeros(257, dtype=bool)
    m = state.mask()
    # NumPy boilerplate: unpack the bitmask into booleans to index the logits.
    allowed[:state.mask_bit_len] = np.unpackbits(m.astype('<u4').view(np.uint8), bitorder='little')[:state.mask_bit_len].astype(bool)
    allowed[EOS] = state.is_accepting()
    if not allowed.any():
        raise RuntimeError("no allowed token and cannot stop")
    token = int(np.argmax(np.where(allowed, logits, -np.inf)))
    if token == EOS:
        break
    if not state.commit_token(token):
        raise RuntimeError("allowed token rejected")
    output.append(token)

if not state.is_accepting():
    raise RuntimeError("incomplete output")
print(bytes(output).decode())  # e.g. {"color": "blue"}
```

## Learn more

[Full guide](https://github.com/IsaacBreen/glrmask/blob/main/GUIDE.md) · [API reference](https://glrmask.com/api/python/)
"""
from __future__ import annotations
from collections.abc import Mapping
from typing import Any, ClassVar, Iterable, Literal, final
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
    @staticmethod
    def from_glrm(source: str, /) -> Grammar:
        """Parses [GLRM](https://github.com/IsaacBreen/glrmask/blob/main/docs/glrm-format.md) source."""
        ...
    @staticmethod
    def from_lark(source: str, /) -> Grammar:
        """Parses [Lark](https://github.com/IsaacBreen/glrmask/blob/main/GUIDE.md#grammar-formats) source."""
        ...
    @staticmethod
    def from_json_schema(schema: str | bool | dict[str, Any], /) -> Grammar:
        """Parses a [JSON Schema](https://github.com/IsaacBreen/glrmask/blob/main/GUIDE.md#grammar-formats), given as JSON text or a dict."""
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
