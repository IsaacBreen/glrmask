"""GLRMask is a grammar-constrained decoding library for high-throughput LLM serving. It is optimized for fast mask computation with extremely low tail latency, even for complex grammars.

GLRMask supports general context-free grammars, including ambiguous ones. It accepts JSON Schema, Lark and [GLRM](https://github.com/IsaacBreen/glrmask/blob/main/docs/glrm-format.md).

## Installation

```bash
python -m pip install glrmask
```

## Usage

GLRMask has four core types:

- `Grammar`: a grammar.
- `Module`: a grammar that can be compiled before its externs are bound, so the final compile only links the parts.
- `Constraint`: a grammar compiled for a vocabulary, ready to use.
- `ConstraintState`: the state of one generated sequence.

## Quickstart

```python
import glrmask

grammar = glrmask.Grammar.from_lark('''
    start: "hello" " " "world"
''')

# A toy vocabulary of token ID -> bytes. With a real model, build it from the tokenizer.
vocab = glrmask.Vocab({0: b"hello", 1: b" ", 2: b"world"})

constraint = grammar.compile(vocab)
state = constraint.start()

# Each mask says which tokens may come next. Apply it to the logits before sampling.
assert state.mask(dtype=bool).tolist() == [True, False, False]  # only "hello"
assert state.commit_token(0)
assert state.mask(dtype=bool).tolist() == [False, True, False]  # only " "
assert state.commit_token(1)
assert state.commit_token(2)
assert state.is_accepting()
```

## Learn more

[Full guide](https://github.com/IsaacBreen/glrmask/blob/main/GUIDE.md) · [API reference](https://glrmask.com/api/python/)
"""
from __future__ import annotations
from collections.abc import Iterator, Mapping, KeysView, ValuesView, ItemsView
from types import ModuleType
from typing import Any, ClassVar, Iterable, Literal, Sequence, TypeVar, final, overload
import numpy as np
from numpy.typing import NDArray

_T = TypeVar("_T")

_internal: ModuleType

__version__: str
"""The GLRMask version, for example "1.0.0"."""
__all__: list[str] = [
    "Vocab",
    "Grammar",
    "Expr",
    "TokenSet",
    "Module",
    "Constraint",
    "ConstraintState",
    "Externs",
    "Extern",
    "MaskStrategy",
    "BoundaryMaskStrategy",
    "GLRMaskError",
    "__version__",
]

@final
class Vocab:
    """A vocabulary: the bytes of each token ID."""
    def __init__(self, id_to_bytes: Mapping[int, bytes], /) -> None:
        """Creates a vocabulary from an `{id: bytes}` mapping."""
        ...

@final
class Grammar:
    """A grammar."""
    @staticmethod
    def from_glrm(source: str, /) -> Grammar:
        """Creates a grammar from GLRM source."""
        ...
    @staticmethod
    def from_lark(source: str, /) -> Grammar:
        """Creates a grammar from Lark source."""
        ...
    @staticmethod
    def from_json_schema(schema: Any, /) -> Grammar:
        """Creates a grammar from a JSON Schema, as JSON text or a dict."""
        ...
    @staticmethod
    def from_expr(expr: Expr, /) -> Grammar:
        """Creates a grammar from an expression."""
        ...
    @property
    def externs(self) -> Externs:
        """The grammar's externs."""
        ...
    @overload
    def bind(self, name: str, value: Grammar | Expr, /) -> None:
        """Binds the extern `name` to `value`. `bind(a=x, b=y)` binds several at once; if one fails, none is bound.

        Raises `GLRMaskError` if a name is undeclared or already bound, or a value is the wrong kind.
        """
        ...
    @overload
    def bind(self, **bindings: Grammar | Expr) -> None:
        """Binds the extern `name` to `value`. `bind(a=x, b=y)` binds several at once; if one fails, none is bound.

        Raises `GLRMaskError` if a name is undeclared or already bound, or a value is the wrong kind.
        """
        ...
    def compile(
        self,
        vocab: Vocab,
        *,
        mask_strategy: MaskStrategy | None = None,
        boundary_mask_strategy: BoundaryMaskStrategy | None = None,
    ) -> Constraint:
        """Compiles the grammar for `vocab`.

        Raises `GLRMaskError` if an extern is unbound. See `MaskStrategy` for the keyword arguments.
        """
        ...
    def precompile(
        self,
        vocab: Vocab,
        *,
        mask_strategy: MaskStrategy | None = None,
        boundary_mask_strategy: BoundaryMaskStrategy | None = None,
    ) -> Module:
        """Compiles what it can for `vocab` now, even with externs unbound, and returns a `Module` that keeps the work.

        See `MaskStrategy` for the keyword arguments.
        """
        ...
    def copy(self) -> Grammar:
        """A copy of the grammar."""
        ...
    def __copy__(self) -> Grammar:
        """Same as `copy()`."""
        ...
    def __deepcopy__(self, memo: dict[int, Any]) -> Grammar:
        """Same as `copy()`."""
        ...

@final
class Expr:
    """A grammar expression."""
    @staticmethod
    def literal(value: str | bytes, /) -> Expr:
        """Matches exactly `value`. A `str` is UTF-8 encoded."""
        ...
    @staticmethod
    def sequence(*parts: Grammar | Module | Expr | TokenSet) -> Expr:
        """Matches `parts` in order.

        Raises `GLRMaskError` if a part has unbound externs.
        """
        ...
    @staticmethod
    def choice(*parts: Grammar | Module | Expr | TokenSet) -> Expr:
        """Matches any one of `parts`.

        Raises `GLRMaskError` if a part has unbound externs.
        """
        ...

@final
class TokenSet:
    """A set of token IDs."""
    def __new__(cls, ids: Iterable[int], /) -> TokenSet:
        """Creates a set from `ids`."""
        ...
    @staticmethod
    def from_ranges(ranges: Iterable[range], /) -> TokenSet:
        """Creates a set from `range` objects, which must have step 1."""
        ...
    def __repr__(self) -> str:
        """An expression that rebuilds the set, or a summary for large sets."""
        ...
    def __eq__(self, other: object, /) -> bool:
        """Whether both sets contain the same IDs."""
        ...

@final
class Module:
    """A grammar that can be compiled before its externs are bound. Binding them later only links the compiled parts, so the final compile is fast."""
    def __new__(cls, source: Grammar, /) -> Module:
        """Creates a module from a grammar."""
        ...
    @overload
    def bind(self, name: str, value: Grammar | Module | Expr | TokenSet, /) -> None:
        """Binds the extern `name` to `value`. `bind(a=x, b=y)` binds several at once; if one fails, none is bound.

        Raises `GLRMaskError` if a name is undeclared or already bound, or a value is the wrong kind.
        """
        ...
    @overload
    def bind(self, **bindings: Grammar | Module | Expr | TokenSet) -> None:
        """Binds the extern `name` to `value`. `bind(a=x, b=y)` binds several at once; if one fails, none is bound.

        Raises `GLRMaskError` if a name is undeclared or already bound, or a value is the wrong kind.
        """
        ...
    @property
    def externs(self) -> Externs:
        """The module's externs."""
        ...
    def precompile(
        self,
        vocab: Vocab,
        *,
        mask_strategy: MaskStrategy | None = None,
        boundary_mask_strategy: BoundaryMaskStrategy | None = None,
    ) -> None:
        """Compiles what it can for `vocab` now, even with externs unbound, so later compiles are faster.

        See `MaskStrategy` for the keyword arguments.
        """
        ...
    def compile(
        self,
        vocab: Vocab,
        *,
        mask_strategy: MaskStrategy | None = None,
        boundary_mask_strategy: BoundaryMaskStrategy | None = None,
    ) -> Constraint:
        """Compiles the module for `vocab`, reusing any precompiled work.

        Raises `GLRMaskError` if an extern is unbound or the module was precompiled for another vocabulary. See `MaskStrategy` for the keyword arguments.
        """
        ...
    def copy(self) -> Module:
        """A copy of the module."""
        ...
    def save(self) -> bytes:
        """The module as bytes, with its precompiled work but not the vocabulary. May not load in another GLRMask version."""
        ...
    @staticmethod
    def load(data: bytes, /) -> Module:
        """Loads a module saved with `save()`."""
        ...
    def __copy__(self) -> Module:
        """Same as `copy()`."""
        ...
    def __deepcopy__(self, memo: dict[int, Any]) -> Module:
        """Same as `copy()`."""
        ...

@final
class Constraint:
    """A grammar compiled for a vocabulary, ready to use. Call `start()` for each sequence you generate."""
    def start(self) -> ConstraintState:
        """A new state with nothing committed."""
        ...
    def mask_bit_len(self) -> int:
        """The number of bits in the mask: one more than the largest token ID in the vocabulary or a bound token set."""
        ...
    def save(self) -> bytes:
        """The constraint as bytes, without the vocabulary. May not load in another GLRMask version."""
        ...
    @staticmethod
    def load(data: bytes, vocab: Vocab, /) -> Constraint:
        """Loads a constraint saved with `save()`. Raises `GLRMaskError` if `vocab` isn't the one it was compiled for."""
        ...

@final
class ConstraintState:
    """The state of one generated sequence."""
    def mask(self, *, dtype: type[np.uint32] | type[bool] = np.uint32) -> NDArray[np.uint32] | NDArray[np.bool_]:
        """The allowed tokens: packed `uint32` words by default, where token `id` is bit `id % 32` of word `id // 32`. With `dtype=bool`, one boolean per token ID, `mask_bit_len()` long."""
        ...
    def fill_mask(self, bitmask: NDArray[np.uint32], /) -> None:
        """Writes the packed mask into `bitmask`, a writable, contiguous 1-D `uint32` array of at least `(mask_bit_len() + 31) // 32` words. Extra words are zeroed.

        Releases the GIL, so other threads can fill masks for other states at the same time. Don't touch `bitmask` from another thread until it returns. Raises `TypeError` for an unsuitable array, or `GLRMaskError` if it's too short.
        """
        ...
    def mask_bit_len(self) -> int:
        """The number of bits in the mask: one more than the largest token ID in the vocabulary or a bound token set."""
        ...
    def commit_token(self, token_id: int, /) -> bool:
        """Commits a token. If it isn't allowed, returns `False` and terminates the state."""
        ...
    def commit_tokens(self, token_ids: Sequence[int], /) -> int:
        """Commits tokens in order and returns how many were committed. If that's fewer than `len(token_ids)`, the next token wasn't allowed and the state is terminated."""
        ...
    def commit_bytes(self, data: bytes, /) -> bool:
        """Commits raw bytes. If they aren't allowed, returns `False` and terminates the state."""
        ...
    def commit_forced_tokens(self) -> list[int]:
        """Commits the tokens for what must come next, taking the longest token at each step, and returns them. Stops before anything that could vary, or after 1,000 tokens."""
        ...
    def is_accepting(self) -> bool:
        """Whether what's committed so far is a complete match. When true, allow EOS in your sampler; don't commit it."""
        ...
    def is_terminated(self) -> bool:
        """Whether nothing more can be committed."""
        ...
    def copy(self) -> ConstraintState:
        """A copy of the state."""
        ...
    def __copy__(self) -> ConstraintState:
        """Same as `copy()`."""
        ...
    def __deepcopy__(self, memo: dict[int, Any]) -> ConstraintState:
        """Same as `copy()`."""
        ...

@final
class Externs(Mapping[str, "Extern"]):
    """A read-only mapping from extern name to `Extern`, in declaration order."""
    @property
    def unbound(self) -> Externs:
        """The externs not yet bound."""
        ...
    def __getitem__(self, key: str, /) -> Extern:
        """The extern named `key`. Raises `KeyError` if there is none."""
        ...
    def __iter__(self) -> Iterator[str]:
        """The extern names, in declaration order."""
        ...
    def __len__(self) -> int:
        """The number of externs."""
        ...
    @overload
    def get(self, key: str, /) -> Extern | None:
        """The extern named `key`, or `default` if there is none."""
        ...
    @overload
    def get(self, key: str, default: Extern | _T | None, /) -> Extern | _T: ...
    def keys(self) -> KeysView[str]:
        """The extern names, in declaration order."""
        ...
    def values(self) -> ValuesView[Extern]:
        """The externs, in declaration order."""
        ...
    def items(self) -> ItemsView[str, Extern]:
        """`(name, extern)` pairs, in declaration order."""
        ...
    def __repr__(self) -> str:
        """A representation for debugging."""
        ...
    def __eq__(self, other: object, /) -> bool:
        """Whether both list the same externs in the same order."""
        ...

@final
class Extern:
    """A named placeholder in a grammar."""
    @property
    def name(self) -> str:
        """The extern's name."""
        ...
    @property
    def kind(self) -> Literal["token", "grammar"]:
        """What the extern can be bound to: `"token"` or `"grammar"`."""
        ...
    def __repr__(self) -> str:
        """A representation for debugging."""
        ...
    def __eq__(self, other: object, /) -> bool:
        """Whether both have the same name and kind."""
        ...

@final
class MaskStrategy:
    """A preference for faster compiling or faster masks."""
    AUTO: ClassVar[MaskStrategy]
    """Let the library choose. The default."""
    FAST_COMPILE: ClassVar[MaskStrategy]
    """Prefer lower preparation cost, even if masks cost more."""
    FAST_MASKS: ClassVar[MaskStrategy]
    """Prefer lower mask cost, even if preparation costs more."""

@final
class BoundaryMaskStrategy:
    """The same preference, for masks where one grammar meets another at an extern."""
    AUTO: ClassVar[BoundaryMaskStrategy]
    """Let the library choose. The default."""
    FAST_COMPILE: ClassVar[BoundaryMaskStrategy]
    """Prefer lower boundary preparation cost."""
    FAST_MASKS: ClassVar[BoundaryMaskStrategy]
    """Prefer lower boundary mask cost."""

class GLRMaskError(ValueError):
    """Raised when GLRMask can't do what was asked."""
    kind: str
    """What went wrong: one of `"syntax"`, `"invalid_input"`, `"compilation"`, `"resource_limit"`, `"strategy_unsupported"`, `"serialization"`, `"internal"`. New kinds may be added in a minor release."""
